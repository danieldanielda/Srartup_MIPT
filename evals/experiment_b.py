"""
Gold Apple RAG Evaluation Script — Full Quality Metrics
LLM-as-Judge + Embeddings + BERTScore
(Версия с полным переходом на OpenAI Client)
"""

import json
import os
import requests
import numpy as np
import time
import logging
from typing import List, Dict, Tuple
from pathlib import Path

# OpenAI Client import
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

try:
    from bert_score import score as bert_score_calc
except ImportError:
    print("Warning: bert-score not installed. Install with: pip install bert-score")
    bert_score_calc = None

from config import EvalSettings
settings = EvalSettings()

# ==============================================================================
# LOGGING & CONFIG
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eval_run.log", encoding="utf-8", mode="w")
    ]
)
logger = logging.getLogger(__name__)

# API endpoints
API_BASE_URL = "http://195.209.219.147:8457/api/v1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Пути к данным (относительные)
DATASET_H1_PATH = PROJECT_ROOT / "evals" / "eval_dataset_nl_queries.jsonl"
GOLDAPPLE_DB_PATH = PROJECT_ROOT / "data" / "parser" / "goldapple_dataset.json"

# Embeddings API (остается requests, так как это не OpenAI-compatible)
EMB_API_URL = "http://195.209.219.147:8080/embed"
EMB_API_KEY = "secret"
EMB_MODEL = "BAAI/bge-m3"

# System prompts
SYSTEM_PROMPT_NO_CONTEXT = """Ты — экспертный ассистент магазина косметики Gold Apple.
Отвечай точно, кратко и по делу. Если не знаешь ответа — так и скажи.
Не выдумывай факты о товарах, составах, ценах или наличии.
Твой ответ должен быть полезным, но строго в рамках твоих знаний."""

SYSTEM_PROMPT_RAG = """Ты — экспертный ассистент магазина косметики Gold Apple.
Используй предоставленный контекст (информацию о товарах) для формирования ответа.
Если в контексте нет информации для ответа — честно скажи об этом.
Не добавляй факты, которых нет в предоставленных источниках."""

# ==============================================================================
# OPENAI CLIENT INIT
# ==============================================================================

# Инициализация клиента для всех вызовов LLM (Judge + Baseline)
# ВАЖНО: settings.model_api должен быть базовым URL, например: "https://routerai.ru/api/v1"
# Клиент сам добавит /chat/completions при вызове
llm_client = OpenAI(
    api_key=settings.model_api_key,
    base_url=settings.model_api,
    timeout=120,
    max_retries=3
)

# ==============================================================================
# 1. LLM-as-Judge Logic (Factual Accuracy & Hallucinations)
# ==============================================================================

def decompose_answer_into_claims(answer: str) -> List[str]:
    """Разбивает ответ на атомарные утверждения (предложения) для проверки."""
    if not answer:
        return []
    # Простая эвристика: разделение по точкам, фильтрация коротких/мусорных фраз
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 15]
    cleaned = [s for s in sentences if not s.strip().startswith(('http', 'www', '*', '-'))]
    return cleaned


def call_llm_judge(claim: str, reference_text: str) -> str:
    """Отправляет утверждение Судье (LLM) через OpenAI-клиент."""
    prompt = f"""
    Ты — строгий эксперт-аудитор фактов.
    Твоя задача: проверить, подтверждается ли Утверждение (Claim) предоставленным Контекстом (Reference).
    
    Reference (Факты из базы знаний):
    "{reference_text}"
    
    Claim (Утверждение из ответа бота):
    "{claim}"
    
    Правила:
    1. Если Claim полностью поддерживается фактами из Reference -> верни SUPPORTED.
    2. Если Claim противоречит фактам из Reference -> верни CONTRADICTED.
    3. Если в Reference нет информации для проверки Claim -> верни UNVERIFIED.
    
    Ответь ТОЛЬКО одним словом: SUPPORTED, CONTRADICTED или UNVERIFIED.
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "Отвечай строго по инструкции. Только одно слово."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10  # Достаточно для одного слова
        )
        
        verdict = response.choices[0].message.content.strip().upper()
        
        if "SUPPORTED" in verdict:
            return "SUPPORTED"
        if "CONTRADICTED" in verdict:
            return "CONTRADICTED"
        return "UNVERIFIED"
        
    except APITimeoutError as e:
        logger.error(f"Judge timeout: {e}")
        return "ERROR"
    except APIConnectionError as e:
        logger.error(f"Judge connection error: {e}")
        return "ERROR"
    except APIError as e:
        status = getattr(e, 'http_status', 'N/A')
        logger.error(f"Judge API error {status}: {e}")
        return "ERROR"
    except Exception as e:
        logger.error(f"Judge unexpected error: {e}", exc_info=True)
        return "ERROR"


def calculate_factual_accuracy_metrics(claims: List[str], reference_text: str) -> Dict[str, float]:
    """Считает Factual Accuracy и Hallucination Rate через LLM-as-Judge."""
    if not claims:
        return {"factual_accuracy": 0.0, "hallucination_rate": 0.0, "total_claims": 0}
    
    results = []
    for claim in claims:
        verdict = call_llm_judge(claim, reference_text)
        results.append(verdict)
        logger.debug(f"  Claim: '{claim[:60]}...' -> {verdict}")
    
    total = len(results)
    supported = results.count("SUPPORTED")
    contradicted = results.count("CONTRADICTED")
    
    # Считаем метрики только по проверяемым утверждениям
    checkable = supported + contradicted
    if checkable == 0:
        return {"factual_accuracy": 0.0, "hallucination_rate": 0.0, "total_claims": total}
    
    accuracy = supported / checkable
    hallucination_rate = contradicted / checkable
    
    return {
        "factual_accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "total_claims": total
    }


def get_embeddings_api(texts: List[str]) -> np.ndarray:
    """Получает эмбеддинги через внешний API (BGE-M3) через requests."""
    if not texts:
        return np.array([])
    if isinstance(texts, str):
        texts = [texts]
    
    # Замена пустых строк на пробел
    clean_texts = [t if t and t.strip() else " " for t in texts]
    
    headers = {"Content-Type": "application/json"}
    if EMB_API_KEY:
        headers["Authorization"] = f"Bearer {EMB_API_KEY}"
    
    payload = {"inputs": clean_texts, "model": EMB_MODEL}
    
    try:
        response = requests.post(EMB_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Поддержка разных форматов ответа
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict) and 'embeddings' in data:
            return np.array(data['embeddings'])
        else:
            logger.warning(f"Unknown embedding response format: {type(data)}")
            return np.zeros((len(clean_texts), 1024))
    except Exception as e:
        logger.error(f"Embedding API Error: {e}")
        return np.zeros((len(clean_texts), 1024))


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычисляет косинусное сходство двух векторов."""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def calculate_answer_relevancy_api(question: str, answer: str) -> float:
    """Answer Relevancy: косинусное сходство эмбеддингов вопроса и ответа."""
    q_emb = get_embeddings_api([question])
    a_emb = get_embeddings_api([answer])
    if q_emb.size == 0 or a_emb.size == 0:
        return 0.0
    return calculate_cosine_similarity(q_emb[0], a_emb[0])


def calculate_groundedness_api(answer: str, context_chunks: List[str]) -> float:
    """
    Groundedness: насколько каждое предложение ответа семантически близко
    к какому-либо чанку контекста (retrieved documents).
    """
    if not context_chunks or not answer:
        return 0.0
    
    # Разбиваем ответ на предложения
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 15]
    if not sentences:
        return 0.0
    
    # Получаем эмбеддинги для всех предложений и чанков одним запросом
    all_texts = sentences + context_chunks
    all_embs = get_embeddings_api(all_texts)
    
    if all_embs.size == 0:
        return 0.0
    
    n_sentences = len(sentences)
    sent_embs = all_embs[:n_sentences]
    chunk_embs = all_embs[n_sentences:]
    
    scores = []
    for s_emb in sent_embs:
        # Для каждого предложения ищем макс. сходство с любым чанком
        sims = np.array([calculate_cosine_similarity(s_emb, c_emb) for c_emb in chunk_embs])
        max_sim = np.max(sims) if sims.size > 0 else 0.0
        scores.append(max_sim)
    
    return float(np.mean(scores))


# ==============================================================================
# 3. BERTScore (Semantic Similarity)
# ==============================================================================

def calculate_bert_score_safe(candidates: List[str], references: List[str]) -> Dict[str, float]:
    """BERTScore F1 с обработкой ошибок."""
    if not bert_score_calc or not candidates or not references:
        return {"bert_f1": 0.0}
    try:
        P, R, F1 = bert_score_calc(
            candidates, references,
            lang="ru",
            verbose=False,
            device='cpu'
        )
        return {"bert_f1": F1.mean().item()}
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"bert_f1": 0.0}


# ==============================================================================
# 4. Data Loading & Real API Calls
# ==============================================================================

def load_goldapple_db() -> Dict[str, dict]:
    """Загружает базу товаров в словарь по article/sku."""
    if not os.path.exists(GOLDAPPLE_DB_PATH):
        logger.warning(f"DB file not found: {GOLDAPPLE_DB_PATH}")
        return {}
    
    with open(GOLDAPPLE_DB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    db_index = {}
    for item in data:
        key = item.get('article') or item.get('sku')
        if key:
            db_index[str(key)] = item
    return db_index


def call_baseline_llm(query: str) -> str:
    """
    Вызов monolithic LLM без retrieval через OpenAI-клиент.
    """
    try:
        response = llm_client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_NO_CONTEXT},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except APITimeoutError as e:
        logger.error(f"Baseline LLM timeout: {e}")
        return "Ошибка: таймаут генерации ответа."
    except APIConnectionError as e:
        logger.error(f"Baseline LLM connection error: {e}")
        return "Ошибка: нет соединения с сервером."
    except APIError as e:
        logger.error(f"Baseline LLM API error: {e}")
        return f"Ошибка API: {e}"
    except Exception as e:
        logger.error(f"Baseline LLM unexpected error: {e}", exc_info=True)
        return "Ошибка генерации ответа."


def call_rag_system(query: str) -> Tuple[str, List[str]]:
    """
    Вызов RAG-системы через эндпоинт: POST /api/v1/rag/ask
    (Этот вызов остаётся на requests, так как это кастомный FastAPI, а не OpenAI-compatible)
    
    Ожидает ответ по схеме ModelResponse:
    {
        "response": str,              # ← текст ответа
        "source_nodes": List[Dict],   # ← ноды с метаданными
        "tokens": Dict,
        "metadata": Dict
    }
    
    Возвращает: (текст ответа, список текстовых чанков контекста для Groundedness)
    """
    # Правильный путь к эндпоинту
    url = f"{API_BASE_URL}/rag/ask"
    
    # Минимальный payload: только query
    payload = {"query": query}
    
    headers = {"Content-Type": "application/json"}
    if hasattr(settings, 'model_api_key') and settings.model_api_key:
        headers["Authorization"] = f"Bearer {settings.model_api_key}"
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        # === 1. Извлекаем текст ответа ===
        # По схеме: поле называется "response"
        rag_answer = data.get("response", "")
        
        # === 2. Извлекаем текстовые чанки из source_nodes для Groundedness ===
        context_chunks = []
        source_nodes = data.get("source_nodes", [])
        
        if isinstance(source_nodes, list):
            for node in source_nodes:
                if isinstance(node, dict):
                    # Приоритет 1: прямой текст в ноде
                    node_text = node.get("text") or node.get("content")
                    
                    # Приоритет 2: текст из метаданных (если нода — ссылка на товар)
                    if not node_text:
                        metadata = node.get("metadata", {}) or {}
                        # Собираем информативные поля из карточки товара
                        parts = []
                        if metadata.get("title"):
                            parts.append(metadata["title"])
                        if metadata.get("description"):
                            parts.append(metadata["description"])
                        if metadata.get("ingredients"):
                            parts.append(f"Состав: {metadata['ingredients']}")
                        if metadata.get("brand"):
                            parts.append(f"Бренд: {metadata['brand']}")
                        node_text = " | ".join(parts)
                    
                    # Приоритет 3: если есть article/sku, подгружаем полное описание из локальной DB
                    if not node_text:
                        item_id = metadata.get("article") or metadata.get("sku") or node.get("article") or node.get("sku")
                        if item_id:
                            db = load_goldapple_db()
                            product = db.get(str(item_id))
                            if product:
                                node_text = f"{product.get('title', '')}. {product.get('description', '')}. Состав: {product.get('ingredients', '')}"
                    
                    if node_text and len(node_text.strip()) > 20:
                        context_chunks.append(node_text.strip())
        
        # Лог для отладки: сколько чанков извлекли
        logger.debug(f"Extracted {len(context_chunks)} context chunks from source_nodes")
        
        return rag_answer, context_chunks
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error in /rag/ask: {e} | Response: {e.response.text[:300] if e.response else 'N/A'}")
        return "", []
    except Exception as e:
        logger.error(f"RAG API Error: {e}", exc_info=True)
        return "", []


# ==============================================================================
# 5. Main Experiment Runner
# ==============================================================================

def run_experiment_b_full(limit_queries: int = 5):
    """
    Запускает полную оценку качества:
    - Factual Accuracy / Hallucination Rate (LLM-as-Judge)
    - Answer Relevancy / Groundedness (Embeddings)
    - BERTScore F1 (Semantic similarity)
    """
    print("\n=== EXPERIMENT B: FULL QUALITY EVALUATION ===")
    logger.info("Starting full quality evaluation experiment")
    
    if not os.path.exists(DATASET_H1_PATH):
        logger.error(f"Dataset not found: {DATASET_H1_PATH}")
        print(f"ERROR: Dataset H1 not found at {DATASET_H1_PATH}")
        return
    
    # Загружаем датасет
    queries = []
    with open(DATASET_H1_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    db_index = load_goldapple_db()
    test_queries = queries[:limit_queries]
    
    print(f"Loaded {len(queries)} queries, testing on first {len(test_queries)}")
    logger.info(f"Testing on {len(test_queries)} queries")
    
    results = []
    
    for i, q in enumerate(test_queries):
        query_text = q['query']
        gt_ids = q.get('ground_truth_product_ids', [])
        
        print(f"\n[{i+1}/{len(test_queries)}] Query: '{query_text}'")
        logger.info(f"Processing query #{i+1}: {query_text}")
        
        # 1. Получаем ответы от систем
        print("  → Calling RAG system...")
        rag_answer, rag_context = call_rag_system(query_text)
        logger.debug(f"  RAG answer: {rag_answer[:200]}...")
        
        print("  → Calling Baseline LLM...")
        baseline_answer = call_baseline_llm(query_text)
        logger.debug(f"  Baseline answer: {baseline_answer[:200]}...")
        
        # 2. Формируем Reference Text (факты из карточки товара)
        ref_text = ""
        if gt_ids and db_index:
            product = db_index.get(str(gt_ids[0]))
            if product:
                desc = product.get('description', '')
                ing = product.get('ingredients', '')
                name = product.get('title', '')
                ref_text = f"Product: {name}. Description: {desc}. Ingredients: {ing}"
        
        if not ref_text:
            logger.warning("No reference text found for factual metrics")
        
        # 3. Метрики LLM-as-Judge (только для RAG)
        judge_metrics = {"factual_accuracy": 0.0, "hallucination_rate": 0.0, "total_claims": 0}
        if ref_text and rag_answer:
            claims = decompose_answer_into_claims(rag_answer)
            if claims:
                judge_metrics = calculate_factual_accuracy_metrics(claims, ref_text)
        
        # 4. Метрики на эмбеддингах
        rel_rag = calculate_answer_relevancy_api(query_text, rag_answer) if rag_answer else 0.0
        rel_base = calculate_answer_relevancy_api(query_text, baseline_answer) if baseline_answer else 0.0
        ground_rag = calculate_groundedness_api(rag_answer, rag_context) if (rag_answer and rag_context) else 0.0
        
        # 5. BERTScore (семантическая близость к описанию товара)
        bert_rag = {"bert_f1": 0.0}
        bert_base = {"bert_f1": 0.0}
        if ref_text:
            if rag_answer:
                bert_rag = calculate_bert_score_safe([rag_answer], [ref_text])
            if baseline_answer:
                bert_base = calculate_bert_score_safe([baseline_answer], [ref_text])
        
        # Логирование результатов
        print(f"  ✓ Judge Acc: {judge_metrics['factual_accuracy']:.2f} | Halluc: {judge_metrics['hallucination_rate']:.2f}")
        print(f"  ✓ Relevancy: RAG={rel_rag:.3f} | Base={rel_base:.3f}")
        print(f"  ✓ Groundedness: {ground_rag:.3f}")
        print(f"  ✓ BERTScore: RAG={bert_rag['bert_f1']:.3f} | Base={bert_base['bert_f1']:.3f}")
        
        results.append({
            "query": query_text,
            "ground_truth_ids": gt_ids,
            "judge_acc_rag": judge_metrics['factual_accuracy'],
            "hallucination_rate_rag": judge_metrics['hallucination_rate'],
            "total_claims": judge_metrics['total_claims'],
            "relevancy_rag": rel_rag,
            "relevancy_base": rel_base,
            "groundedness_rag": ground_rag,
            "bert_f1_rag": bert_rag['bert_f1'],
            "bert_f1_base": bert_base['bert_f1'],
            "rag_answer": rag_answer,
            "baseline_answer": baseline_answer
        })
        
        # Пауза между запросами
        time.sleep(1)
    
    # === ИТОГОВАЯ ТАБЛИЦА ===
    if results:
        print("\n" + "="*60)
        print("FINAL SUMMARY TABLE")
        print("="*60)
        print(f"{'Metric':<30} | {'RAG Avg':<12} | {'Baseline Avg':<12}")
        print("-"*60)
        
        avg_judge_acc = np.mean([r['judge_acc_rag'] for r in results])
        avg_halluc = np.mean([r['hallucination_rate_rag'] for r in results])
        avg_rel_rag = np.mean([r['relevancy_rag'] for r in results])
        avg_rel_base = np.mean([r['relevancy_base'] for r in results])
        avg_ground = np.mean([r['groundedness_rag'] for r in results])
        avg_bert_rag = np.mean([r['bert_f1_rag'] for r in results])
        avg_bert_base = np.mean([r['bert_f1_base'] for r in results])
        
        print(f"{'Factual Accuracy (Judge)':<30} | {avg_judge_acc:<12.4f} | {'N/A':<12}")
        print(f"{'Hallucination Rate (Judge)':<30} | {avg_halluc:<12.4f} | {'N/A':<12}")
        print(f"{'Answer Relevancy (Embeddings)':<30} | {avg_rel_rag:<12.4f} | {avg_rel_base:<12.4f}")
        print(f"{'Groundedness (Embeddings)':<30} | {avg_ground:<12.4f} | {'N/A':<12}")
        print(f"{'BERTScore F1':<30} | {avg_bert_rag:<12.4f} | {avg_bert_base:<12.4f}")
        print("="*60)
        
        # Сохранение результатов
        output_path = "eval_results_full.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
        print(f"\n✓ Results saved to {output_path}")
        
        # Детальный CSV для анализа
        import csv
        csv_path = "eval_results_detailed.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"✓ Detailed results saved to {csv_path}")


if __name__ == "__main__":
    # Запуск: проверить 5 запросов (изменить limit_queries для полного прогона)
    run_experiment_b_full(limit_queries=5)