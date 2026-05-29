"""
Воспроизводимая оценка RAG системы на датасетах H1 и H2
Поддерживает:
- H1: evals/eval_dataset_nl_queries.jsonl (рекомендации)
- H2: evals/eval_dataset_inci.jsonl (анализ ингредиентов)
- LLM-as-Judge для factual accuracy
- Полная воспроизводимость через seed, temperature=0, кэширование
"""

import json
import os
import requests
import numpy as np
import re
import random
import hashlib
import pickle
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import time
import logging
from dataclasses import dataclass, asdict, field
from collections import defaultdict
# После остальных импортов
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

from config import EvalSettings

def set_seed(seed: int = 42):
    """Фиксирует все random seeds для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class DeterministicCache:
    """Кэш для воспроизводимых результатов"""
    
    def __init__(self, cache_dir: str = "eval_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_hash(self, *args, **kwargs) -> str:
        """Создает детерминированный хеш"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def set(self, key: str, value: Any):
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
    
    def clear(self):
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()


cache = DeterministicCache()

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = EvalSettings()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_H1_PATH = PROJECT_ROOT / "evals" / "eval_dataset_nl_queries.jsonl"
DATASET_H2_PATH = PROJECT_ROOT / "evals" / "eval_dataset_inci.jsonl"
GOLDAPPLE_DB_PATH = PROJECT_ROOT / "data" / "parser" / "goldapple_dataset.json"

API_BASE_URL = "http://195.209.219.147:8490/api/v1"

judge_client = OpenAI(
    api_key=settings.model_api_key,
    base_url=settings.model_api,
    timeout=120,
    max_retries=3
)
# ==============================================================================
# 2. ПАРСИНГ ОТВЕТОВ API
# ==============================================================================

def extract_skus_from_text(text: str) -> List[str]:
    """Извлекает SKU из текста"""
    skus = []
    pattern_marked = r'(?:sku_|article[:\s]*)(\d+)'
    found_marked = re.findall(pattern_marked, text, re.IGNORECASE)
    if found_marked:
        skus.extend(found_marked)
    
    if not skus:
        pattern_digits = r'\b(\d{10,14})\b'
        skus = re.findall(pattern_digits, text)
    
    return sorted(list(set(skus)))  # Сортируем для воспроизводимости


def parse_recommendation_response(response_data: dict) -> List[str]:
    """Парсит ответ API рекомендаций"""
    raw_content = response_data.get("recommendations", "")
    
    if isinstance(raw_content, str):
        try:
            clean_json = raw_content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            
            if isinstance(parsed, list):
                result_skus = []
                for item in parsed:
                    if isinstance(item, dict):
                        val = item.get('sku') or item.get('article') or item.get('id')
                        if val:
                            result_skus.append(str(val))
                    elif isinstance(item, str):
                        result_skus.append(item)
                return sorted(set(result_skus))
        except json.JSONDecodeError:
            pass
    
    return extract_skus_from_text(str(raw_content))


def parse_analysis_response(response_data: dict) -> Dict[str, str]:
    """Парсит ответ API анализа ингредиентов"""
    if isinstance(response_data, dict):
        for key in ['ingredients', 'result', 'analysis', 'classified_ingredients']:
            if key in response_data and isinstance(response_data[key], dict):
                return response_data[key]
        
        exclude_keys = {'status', 'message', 'error', 'summary', 'raw'}
        filtered = {k: v for k, v in response_data.items() if k not in exclude_keys}
        if filtered:
            return filtered
    
    return {}

# ==============================================================================
# 3. LLM-as-Judge ДЛЯ ФАКТУАЛЬНОЙ ТОЧНОСТИ
# ==============================================================================

def decompose_answer_into_claims(answer: str) -> List[str]:
    """
    Разбивает ответ на атомарные утверждения.
    Детерминированная версия.
    """
    if not answer:
        return []
    
    claims = []
    # Разбиваем по предложениям
    sentences = re.split(r'[.!?;]+', answer)
    
    for sentence in sentences:
        cleaned = sentence.strip()
        # Фильтруем короткие предложения и URL
        if len(cleaned) > 15 and not cleaned.startswith(('http', 'www', 'https')):
            # Убираем маркеры списков
            cleaned = cleaned.lstrip('•-*0123456789) ')
            if cleaned:
                claims.append(cleaned)
    
    # Сортируем для детерминизма
    return sorted(set(claims))


def call_llm_judge(claim: str, reference_text: str, seed: int = 42, use_cache: bool = True) -> str:
    """
    Отправляет утверждение LLM-судье через OpenAI-клиент (RouterAI).
    temperature=0 и seed для воспроизводимости.
    """
    cache_key = f"judge_{hashlib.md5(f'{claim}|{reference_text}'.encode()).hexdigest()}"
    
    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            return cached
    
    prompt = f"""Ты — строгий эксперт-аудитор фактов.
Проверь утверждение на соответствие Reference.

Утверждение (Claim): "{claim}"
Reference (Факты): "{reference_text[:2000]}"

Верни ТОЛЬКО одно слово:
SUPPORTED - если утверждение подтверждается Reference
CONTRADICTED - если утверждение противоречит Reference  
UNVERIFIED - если в Reference нет информации для проверки"""
    
    try:
        # Вызов через официальный OpenAI-клиент
        response = judge_client.chat.completions.create(
            model=settings.model_name,  # Важно: формат "vendor/model", например "qwen/qwen3-235b-a22b-2507"
            messages=[
                {"role": "system", "content": "Отвечай строго по инструкции. Только одно слово."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=seed,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=10  # Достаточно для одного слова
        )
        
        verdict = response.choices[0].message.content.strip().upper()
        
        # Нормализация ответа
        if "SUPPORTED" in verdict:
            result = "SUPPORTED"
        elif "CONTRADICTED" in verdict:
            result = "CONTRADICTED"
        else:
            result = "UNVERIFIED"
        
        if use_cache:
            cache.set(cache_key, result)
        
        return result
        
    except APITimeoutError as e:
        logger.error(f"Judge timeout: {e}")
        return "ERROR"
    except APIConnectionError as e:
        logger.error(f"Judge connection error: {e}")
        return "ERROR"
    except APIError as e:
        # Ловим 400, 401, 429, 500 и т.д.
        status = getattr(e, 'http_status', 'N/A')
        body = getattr(e, 'body', 'N/A')
        logger.error(f"Judge API error {status}: {body}")
        return "ERROR"
    except Exception as e:
        logger.error(f"Judge unexpected error: {e}", exc_info=True)
        return "ERROR"

def evaluate_with_judge(rag_answer: str, gt_product: dict, seed: int = 42) -> Dict[str, float]:
    """
    Оценивает ответ RAG через LLM-судью на основе GT продукта
    """
    if not rag_answer or not gt_product:
        return {
            "factual_accuracy": 0.0,
            "hallucination_rate": 0.0,
            "total_claims": 0,
            "supported": 0,
            "contradicted": 0,
            "unverified": 0
        }
    
    # Формируем reference текст из GT продукта
    reference_text = f"{gt_product.get('title', '')}. {gt_product.get('description', '')}. Состав: {gt_product.get('ingredients', '')}"
    
    # Разбиваем ответ на утверждения
    claims = decompose_answer_into_claims(rag_answer)
    
    if not claims:
        return {
            "factual_accuracy": 0.0,
            "hallucination_rate": 0.0,
            "total_claims": 0,
            "supported": 0,
            "contradicted": 0,
            "unverified": 0
        }
    
    # Проверяем каждое утверждение
    results = []
    for claim in claims:
        verdict = call_llm_judge(claim, reference_text, seed=seed)
        results.append(verdict)
    
    total = len(results)
    supported = results.count("SUPPORTED")
    contradicted = results.count("CONTRADICTED")
    unverified = results.count("UNVERIFIED")
    
    checkable = supported + contradicted
    accuracy = supported / checkable if checkable > 0 else 0.0
    hallucination_rate = contradicted / checkable if checkable > 0 else 0.0
    
    return {
        "factual_accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "total_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "unverified": unverified
    }

# ==============================================================================
# 4. API ВЫЗОВЫ RAG СИСТЕМЫ
# ==============================================================================

def load_goldapple_db():
    """Загружает базу данных GoldApple для получения информации о продуктах"""
    if not os.path.exists(GOLDAPPLE_DB_PATH):
        logger.warning(f"{GOLDAPPLE_DB_PATH} not found")
        return {}
    
    with open(GOLDAPPLE_DB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    db_index = {}
    for item in data:
        # Пробуем разные ключи
        key = item.get('article') or item.get('sku') or item.get('id')
        if key:
            db_index[str(key)] = item
    
    return db_index


def call_rag_ask(query: str, system_prompt: str = "system_common_prompt") -> Tuple[str, List[str]]:
    """
    Универсальный вызов к эндпоинту /rag/ask.
    Возвращает: (текст ответа, список retrieved article/sku из metadata)
    """
    url = f"{API_BASE_URL}/recommend_products"
    
    # Payload соответствует схеме Query в бэкенде
    payload = {"query": query}
    
    headers = {"Content-Type": "application/json"}
    if hasattr(settings, 'model_api_key') and settings.model_api_key:
        headers["Authorization"] = f"Bearer {settings.model_api_key}"
    
    try:
        # system_prompt передаётся как query-параметр (если бэкенд так ожидает)
        # Или можно добавить в payload, если схема Query поддерживает это поле
        response = requests.post(
            url, 
            json=payload, 
            headers=headers, 
            params={"system_prompt": system_prompt} if system_prompt else None,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        # === Парсинг ответа по схеме ModelResponse ===
        # 1. Текст ответа
        rag_answer = data.get("response", "")
        
        # 2. Извлечение article/sku из source_nodes
        retrieved_ids = []
        source_nodes = data.get("source_nodes", [])
        
        if isinstance(source_nodes, list):
            for node in source_nodes:
                if isinstance(node, dict):
                    # Ищем article/sku в metadata (основной случай)
                    metadata = node.get("metadata") or {}
                    item_id = (
                        metadata.get("article") or 
                        metadata.get("sku") or 
                        metadata.get("product_id") or
                        # Fallback: если вдруг в корне ноды
                        node.get("article") or 
                        node.get("sku")
                    )
                    if item_id:
                        retrieved_ids.append(str(item_id))
        
        # Fallback: парсим SKU из текста ответа, если metadata пустая
        if not retrieved_ids and rag_answer:
            extracted = extract_skus_from_text(rag_answer)
            if extracted:
                retrieved_ids = extracted
                logger.debug(f"Fallback: extracted {len(retrieved_ids)} SKUs from answer text")
        
        # Уникализация + сортировка для воспроизводимости
        retrieved_ids = sorted(list(set(retrieved_ids)))
        
        return rag_answer, retrieved_ids
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error in /rag/ask: {e} | Response: {e.response.text[:300] if e.response else 'N/A'}")
        return "", []
    except Exception as e:
        logger.error(f"RAG API Error: {e}", exc_info=True)
        return "", []


def call_recommendation_api(query: str) -> Tuple[str, List[str]]:
    """
    H1: Вызов эндпоинта рекомендаций
    POST /recommend_products
    Возвращает: (текст ответа, список retrieved SKU)
    """
    url = f"{API_BASE_URL}/recommend_products"
    payload = {"query": query}
    
    headers = {"Content-Type": "application/json"}
    if hasattr(settings, 'model_api_key') and settings.model_api_key:
        headers["Authorization"] = f"Bearer {settings.model_api_key}"
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        recommendations = data.get("recommendations", [])
        if isinstance(recommendations, list):
            skus = []
            for item in recommendations:
                if isinstance(item, dict):
                    sku = item.get('sku') or item.get('article') or item.get('id')
                    if sku:
                        skus.append(str(sku))
                elif isinstance(item, str):
                    skus.append(item)
            return data.get("response", ""), sorted(set(skus))
        
        # Fallback: парсим SKU из текста
        text = data.get("response", "")
        return text, extract_skus_from_text(text)
        
    except Exception as e:
        logger.error(f"Recommendation API Error: {e}")
        return "", []


def call_analysis_api(inci_list: List[str]) -> Dict[str, str]:
    """
    H2: Вызов эндпоинта анализа ингредиентов
    POST /analyze_product
    Возвращает: словарь {ингредиент: категория}
    """
    url = f"{API_BASE_URL}/analyze_product"
    payload = {"inci_list": inci_list} 
    
    headers = {"Content-Type": "application/json"}
    if hasattr(settings, 'model_api_key') and settings.model_api_key:
        headers["Authorization"] = f"Bearer {settings.model_api_key}"
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        # Парсинг: ищем поле с классификацией
        result = (
            data.get("classified_ingredients") or 
            data.get("analysis") or 
            data.get("result") or 
            data.get("ingredients")
        )
        
        if isinstance(result, dict):
            # Нормализация ключей и значений
            return {
                str(k).lower().strip(): str(v).lower().strip() 
                for k, v in result.items()
            }
        
        logger.warning(f"Unexpected response format from /analyze_product: {data}")
        return {}
        
    except Exception as e:
        logger.error(f"Analysis API Error: {e}")
        return {}

# ==============================================================================
# 5. МЕТРИКИ ДЛЯ ДАТАСЕТОВ
# ==============================================================================

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """Вычисляет Recall@K для H1 датасета"""
    if not ground_truth_ids:
        return 0.0
    
    # Приводим к строкам для сравнения
    retrieved_str = [str(x) for x in retrieved_ids[:k]]
    gt_str = [str(x) for x in ground_truth_ids]
    
    relevant = set(retrieved_str).intersection(set(gt_str))
    return len(relevant) / len(gt_str)


def calculate_accuracy_for_h2(predictions: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    """Вычисляет accuracy для H2 датасета (классификация ингредиентов)"""
    if not ground_truth:
        return 0.0
    
    correct = 0
    total = len(ground_truth)
    
    for ingredient, true_category in ground_truth.items():
        pred_category = predictions.get(ingredient)
        if pred_category and str(pred_category).lower().strip() == str(true_category).lower().strip():
            correct += 1
    
    return correct / total if total > 0 else 0.0

# ==============================================================================
# 6. ОСНОВНЫЕ ЭКСПЕРИМЕНТЫ
# ==============================================================================

def run_h1_experiment(limit_queries: int = None, seed: int = 42, use_judge: bool = False):
    """
    Запуск эксперимента H1 на датасете eval/eval_dataset_nl_queries.jsonl
    Оценивает:
    - Recall@5
    - Factual Accuracy (через LLM-as-Judge)
    - Hallucination Rate
    """
    print("\n" + "="*80)
    print(f"📊 EXPERIMENT H1: RECOMMENDATIONS EVALUATION")
    print(f"   Seed: {seed} | Judge: {'ON' if use_judge else 'OFF'}")
    print("="*80)
    
    # Проверяем наличие датасета
    if not os.path.exists(DATASET_H1_PATH):
        print(f"❌ Dataset not found: {DATASET_H1_PATH}")
        return None
    
    # Загружаем датасет
    queries = []
    with open(DATASET_H1_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    # Сортируем для воспроизводимости
    queries = sorted(queries, key=lambda x: x.get('query', ''))
    
    if limit_queries:
        queries = queries[:limit_queries]
    
    # Загружаем базу продуктов
    db_index = load_goldapple_db()
    
    # Результаты
    results = []
    
    for i, query_data in enumerate(queries):
        query_text = query_data.get('query', '')
        ground_truth_ids = query_data.get('ground_truth_product_ids', [])
        
        print(f"\n[{i+1}/{len(queries)}] Query: {query_text[:80]}...")
        print(f"   GT Products: {ground_truth_ids}")
        
        # Вызываем RAG систему
        rag_answer, retrieved_skus = call_recommendation_api(query_text)
        
        # Считаем Recall@5
        recall = calculate_recall_at_k(retrieved_skus, ground_truth_ids, k=5)
        print(f"   📈 Recall@5: {recall:.4f}")
        print(f"   Retrieved SKUs: {retrieved_skus[:5]}")
        
        # LLM-as-Judge оценка
        judge_metrics = {}
        if use_judge and ground_truth_ids and db_index:
            # Берем первый продукт из GT для оценки
            first_gt_id = str(ground_truth_ids[0])
            gt_product = db_index.get(first_gt_id)
            
            if gt_product:
                judge_metrics = evaluate_with_judge(rag_answer, gt_product, seed=seed)
                print(f"   ⚖️  Factual Accuracy: {judge_metrics['factual_accuracy']:.4f}")
                print(f"   🎭 Hallucination Rate: {judge_metrics['hallucination_rate']:.4f}")
                print(f"   📝 Claims: {judge_metrics['total_claims']} (S:{judge_metrics['supported']}, C:{judge_metrics['contradicted']}, U:{judge_metrics['unverified']})")
            else:
                print(f"   ⚠️  Product {first_gt_id} not found in database")
        
        results.append({
            "query": query_text,
            "ground_truth_ids": ground_truth_ids,
            "retrieved_skus": retrieved_skus,
            "recall_at_5": recall,
            "rag_answer": rag_answer[:500],
            **judge_metrics
        })
        
        time.sleep(1)  # Пауза между запросами
    
    # Агрегируем результаты
    print("\n" + "-"*80)
    print("📊 H1 RESULTS SUMMARY")
    print("-"*80)
    
    avg_recall = np.mean([r['recall_at_5'] for r in results])
    print(f"✅ Mean Recall@5: {avg_recall:.4f} (Target: >= 0.9)")
    
    if use_judge and any(r.get('factual_accuracy', 0) > 0 for r in results):
        valid_judge_results = [r for r in results if r.get('factual_accuracy', 0) > 0]
        if valid_judge_results:
            avg_accuracy = np.mean([r['factual_accuracy'] for r in valid_judge_results])
            avg_hallucination = np.mean([r['hallucination_rate'] for r in valid_judge_results])
            print(f"✅ Mean Factual Accuracy: {avg_accuracy:.4f}")
            print(f"✅ Mean Hallucination Rate: {avg_hallucination:.4f}")
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"h1_results_seed_{seed}_{timestamp}.json"
    
    output = {
        "experiment": "H1",
        "config": {
            "seed": seed,
            "limit_queries": limit_queries,
            "use_judge": use_judge,
            "dataset": DATASET_H1_PATH,
            "timestamp": timestamp
        },
        "summary": {
            "mean_recall_at_5": float(avg_recall),
            "total_queries": len(results)
        },
        "detailed_results": results
    }
    
    if use_judge and any(r.get('factual_accuracy', 0) > 0 for r in results):
        output["summary"]["mean_factual_accuracy"] = float(avg_accuracy)
        output["summary"]["mean_hallucination_rate"] = float(avg_hallucination)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


def run_h2_experiment(limit_products: int = None, seed: int = 42):
    """
    Запуск эксперимента H2 на датасете eval/eval_dataset_inci.jsonl
    Оценивает accuracy классификации ингредиентов
    """
    print("\n" + "="*80)
    print(f"📊 EXPERIMENT H2: INGREDIENTS CLASSIFICATION")
    print(f"   Seed: {seed}")
    print("="*80)
    
    # Проверяем наличие датасета
    if not os.path.exists(DATASET_H2_PATH):
        print(f"❌ Dataset not found: {DATASET_H2_PATH}")
        return None
    
    # Загружаем датасет
    products = []
    with open(DATASET_H2_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                products.append(json.loads(line))
    
    # Сортируем для воспроизводимости
    products = sorted(products, key=lambda x: str(x.get('inci_list', [])))
    
    if limit_products:
        products = products[:limit_products]
    
    # Результаты
    results = []
    all_accuracies = []
    
    for i, product_data in enumerate(products):
        inci_list = product_data.get('inci_list', [])
        ground_truth = product_data.get('ground_truth_categories', {})
        
        print(f"\n[{i+1}/{len(products)}] Product with {len(inci_list)} ingredients...")
        print(f"   INCI: {', '.join(inci_list[:5])}{'...' if len(inci_list) > 5 else ''}")
        
        # Вызываем API анализа
        predictions = call_analysis_api(inci_list)
        
        # Считаем accuracy
        if ground_truth:
            accuracy = calculate_accuracy_for_h2(predictions, ground_truth)
            all_accuracies.append(accuracy)
            print(f"   📈 Accuracy: {accuracy:.4f}")
            print(f"   Predictions: {len(predictions)}/{len(ground_truth)} ingredients classified")
        
        results.append({
            "inci_list": inci_list,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "accuracy": accuracy if ground_truth else 0
        })
        
        time.sleep(1)
    
    # Агрегируем результаты
    print("\n" + "-"*80)
    print("📊 H2 RESULTS SUMMARY")
    print("-"*80)
    
    if all_accuracies:
        mean_accuracy = np.mean(all_accuracies)
        print(f"✅ Mean Accuracy: {mean_accuracy:.4f} (Target: >= 0.9)")
        print(f"   Total products: {len(all_accuracies)}")
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"h2_results_seed_{seed}_{timestamp}.json"
    
    output = {
        "experiment": "H2",
        "config": {
            "seed": seed,
            "limit_products": limit_products,
            "dataset": DATASET_H2_PATH,
            "timestamp": timestamp
        },
        "summary": {
            "mean_accuracy": float(mean_accuracy) if all_accuracies else 0,
            "total_products": len(results)
        },
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


def verify_reproducibility(limit_queries: int = 2, runs: int = 2):
    """
    Проверяет воспроизводимость результатов
    """
    print("\n" + "="*80)
    print("🔬 REPRODUCIBILITY VERIFICATION")
    print("="*80)
    
    seed = 42
    all_runs_results = []
    
    for run_id in range(runs):
        print(f"\n--- RUN {run_id + 1} ---")
        
        # Очищаем кэш
        cache.clear()
        
        # Запускаем H1 с небольшим количеством запросов
        results = run_h1_experiment(
            limit_queries=limit_queries, 
            seed=seed, 
            use_judge=False
        )
        
        if results:
            # Извлекаем ключевые метрики для сравнения
            run_metrics = []
            for r in results:
                run_metrics.append({
                    "query": r['query'],
                    "recall": r['recall_at_5'],
                    "accuracy": r.get('factual_accuracy', 0),
                    "hallucination": r.get('hallucination_rate', 0)
                })
            all_runs_results.append(run_metrics)
    
    # Сравниваем результаты
    if len(all_runs_results) >= 2:
        print("\n" + "-"*80)
        print("VERIFICATION RESULTS")
        print("-"*80)
        
        identical = True
        for i in range(len(all_runs_results[0])):
            for j in range(1, len(all_runs_results)):
                if all_runs_results[0][i] != all_runs_results[j][i]:
                    identical = False
                    print(f"❌ Results differ between Run 1 and Run {j+1}")
                    print(f"   Query: {all_runs_results[0][i]['query']}")
                    print(f"   Run 1: {all_runs_results[0][i]}")
                    print(f"   Run {j+1}: {all_runs_results[j][i]}")
                    break
        
        if identical:
            print("\n✅ SUCCESS: All runs produced IDENTICAL results!")
            print("   The evaluation is fully reproducible.")
        else:
            print("\n⚠️  WARNING: Results differ between runs.")
            print("   Check if API respects temperature=0 and seed parameters.")

# ==============================================================================
# 7. MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reproducible RAG Evaluation on H1/H2 Datasets")
    parser.add_argument("--exp", type=str, choices=["h1", "h2", "both", "verify"], 
                       default="both", help="Experiment to run")
    parser.add_argument("--limit-h1", type=int, default=None, 
                       help="Limit number of queries for H1 (default: all)")
    parser.add_argument("--limit-h2", type=int, default=None,
                       help="Limit number of products for H2 (default: all)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--no-judge", action="store_true",
                       help="Disable LLM-as-Judge for H1")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--verify-runs", type=int, default=2,
                       help="Number of runs for reproducibility verification")
    
    args = parser.parse_args()
    
    # Настройка
    set_seed(args.seed)
    
    if args.no_cache:
        cache.clear()
        cache.get = lambda x: None
        cache.set = lambda x, y: None
    
    print(f"\n🚀 Starting Reproducible Evaluation Framework")
    print(f"   📍 Dataset H1: {DATASET_H1_PATH}")
    print(f"   📍 Dataset H2: {DATASET_H2_PATH}")
    print(f"   🎲 Seed: {args.seed}")
    print(f"   💾 Cache: {'OFF' if args.no_cache else 'ON'}")
    print(f"   🌡️  LLM Temperature: 0.0 (fixed)")
    
    # Запуск экспериментов
    if args.exp == "verify":
        verify_reproducibility(limit_queries=args.limit_h1 or 2, runs=args.verify_runs)
    
    elif args.exp == "h1":
        run_h1_experiment(
            limit_queries=args.limit_h1,
            seed=args.seed,
            use_judge=not args.no_judge
        )
    
    elif args.exp == "h2":
        run_h2_experiment(
            limit_products=args.limit_h2,
            seed=args.seed
        )
    
    elif args.exp == "both":
        run_h1_experiment(
            limit_queries=args.limit_h1,
            seed=args.seed,
            use_judge=not args.no_judge
        )
        print("\n" + "="*80)
        run_h2_experiment(
            limit_products=args.limit_h2,
            seed=args.seed
        )
    
    print("\n✅ Evaluation complete!")