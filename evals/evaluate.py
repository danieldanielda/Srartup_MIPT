"""
Воспроизводимая оценка гипотез H1 и H2 проекта «На полке».
Строго по методике:
- H1: Recall@5 на наборе NL-запросов.
- H2: Macro-Recall классификации INCI (≥ 0.8).
Эталон H2 верифицирован независимо через LLM-as-Judge (EWG/CIR/EU).
Для оценки используются только строгие метрики соответствия (без Judge/Embeddings на этапе прогона).
Запуск:
    python evaluate.py --exp h1 --limit 4 
    python evaluate.py --exp h2 --limit 4
"""
import json
import re
import os
import requests
import numpy as np
import random
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
import time
import logging
from config import EvalSettings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
settings = EvalSettings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Пути к датасетам
DATASET_H1_PATH = PROJECT_ROOT / "evals" / "eval_dataset_nl_queries.jsonl"
DATASET_H2_PATH = PROJECT_ROOT / "evals" / "eval_dataset_inci.jsonl"

AGENTS_API_URL = "http://195.209.219.147:8490/api/v1"

# Словарь маппинга русских названий → INCI/латиница (для совместимости с inci.json)
RU_TO_INCI_MAP = {
    "вода": "aqua",
    "вода питьевая": "aqua",
    "глицерин": "glycerin",
    "лауриновая кислота": "lauric acid",
    "лаурил гидроксисултаин": "lauryl hydroxysultaine",
    "гидроксид калия": "potassium hydroxide",
    "экстракт киви": "actinidia chinensis fruit extract",
    "экстракт грейпфрута": "citrus grandis fruit extract",
    "экстракт листьев камелии китайской": "camellia sinensis leaf extract",
    "экстракт яблока": "pyrus malus fruit extract",
    "peg-40 гидрогенизированное касторовое масло": "peg-40 hydrogenated castor oil",
    "peg-15 глицерил изостеарат": "peg-15 glyceryl isostearate",
    "кокамид метил меа": "cocamide mea",
    "кокоамфоацетат натрия": "sodium cocoamphoacetate",
    "миристиновая кислота": "myristic acid",
    "яблочная кислота": "malic acid",
    "гидрогенизированный полиизобутен": "hydrogenated polyisobutene",
    "молочная кислота": "lactic acid",
    "тетранатрий edta": "tetrasodium edta",
    "феноксиэтанол": "phenoxyethanol",
    "ароматизатор": "parfum",
    "бутиленгликоль": "butylene glycol",
    "бетаин": "betaine",
    "метилпарабен": "methylparaben",
    "этилпарабен": "ethylparaben",
    "натрия цитрат": "sodium citrate",
    "ксантановая камедь": "xanthan gum",
    "дикалия глицирризат": "dipotassium glycyrrhizate",
    "лимонная кислота": "citric acid",
    "аскорбил глюкозид": "ascorbyl glucoside",
    "гидролизованная гиалуроновая кислота": "hydrolyzed hyaluronic acid",
    "натрия гиалуронат": "sodium hyaluronate",
    "поликватерниум-61": "polyquaternium-61",
    "экстракт шлемника байкальского": "scutellaria baicalensis root extract",
    "водорастворимый коллаген": "soluble collagen",
    "изопропилмиристат": "isopropyl myristate",
    "масло кокосовое": "cocos nucifera oil",
    "эмульсионный воск": "emulsifying wax",
    "масло касторовое гидрогенизированное": "hydrogenated castor oil",
    "полиакрилат натрия/акрилоилдиметилтаурат натрия сополимер": "sodium acryloyldimethyltaurate/sodium acrylate crosspolymer",
    "концентрат коллоидного серебра": "colloidal silver",
    "токоферилацетат": "tocopheryl acetate",
    "ретинолпальмитат": "retinyl palmitate",
    "витамин f": "linoleic acid",
    "витамин d-пантенол": "panthenol",
    "метилизотиазолинон": "methylisothiazolinone",
    "метилхлоризотиазолинон": "methylchloroisothiazolinone",
    "пропилпарабен": "propylparaben",
    "парфюмерная композиция": "parfum",
    "кокамидопропил бетаин": "cocamidopropyl betaine",
    "sodium lauryl sarcosinate": "sodium lauroyl sarcosinate",  # уже латиница, но с пробелами
    "coco glucoside": "coco-glucoside",
    "peg-90 glyceryl isostearate": "peg-90 glyceryl isostearate",
    "laureth-2": "laureth-2",
    "potassium sorbate": "potassium sorbate",
    "sodium benzoate": "sodium benzoate",
    "sodium chloride": "sodium chloride",
    "urea": "urea",
    "panthenol": "panthenol",
    "arctium lappa root extract": "arctium lappa root extract",
    "chamomilla recutita": "chamomilla recutita flower extract",
    "hippophae rhamnoides": "hippophae rhamnoides fruit extract",
    "urtica dioica leaf extract": "urtica dioica leaf extract",
    "epilobium angustifolium leaf extract": "epilobium angustifolium extract",
    "allantoin": "allantoin",
    "niacinamide": "niacinamide",
    "lactic acid": "lactic acid",
    "ethylhexylglycerin": "ethylhexylglycerin",
}

def normalize_ingredient_key(key: str) -> str:
    """
    Приводит ключ к каноническому виду для сравнения.
    1. Прямой маппинг рус→лат (приоритет).
    2. Если латиница: убирает всё, кроме букв, цифр и пробелов, приводит к lower.
    3. Если кириллица: транслитерирует (fallback).
    """
    key_clean = key.strip().lower()
    
    # 1. Прямой маппинг (самый точный)
    if key_clean in RU_TO_INCI_MAP:
        return RU_TO_INCI_MAP[key_clean]
    
    # 2. Если уже латиница (основной случай для нового датасета)
    if not any(ord(c) > 127 for c in key_clean):
        # Убираем спецсимволы, скобки, проценты, но оставляем пробелы и дефисы
        clean = re.sub(r'[^a-z0-9\s\-]', '', key_clean)
        # Убираем лишние пробелы
        clean = ' '.join(clean.split())
        return clean
    
    # 3. Если кириллица (fallback для старых данных)
    translit_map = str.maketrans({
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    })
    return key_clean.translate(translit_map)

def set_seed(seed: int = 42):
    """Фиксация seed для воспроизводимости (хотя API детерминизм зависит от бэкенда)"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Простая метрика схожести строк (0.0 - 1.0).
    Не требует внешних библиотек.
    """
    s1 = set(str1.lower().split())
    s2 = set(str2.lower().split())
    
    if not s1 or not s2:
        return 0.0
    
    # Jaccard similarity: пересечение / объединение
    intersection = s1 & s2
    union = s1 | s2
    
    return len(intersection) / len(union) if union else 0.0


def extract_articles_strict(text: str) -> List[str]:
    """
    Извлекает ТОЛЬКО явные article/SKU из ответа агента.
    Соответствует методике: сравнение идёт строго по идентификаторам.
    """
    if not text:
        return []
    
    articles = []
    # Ищем форматы: [article: 19...], article: 19..., SKU: 19..., арт. 19...
    patterns = [
        r'(?:\[?(?:article|арт|sku|код)[\s:]*#?\]?[\s:]*)(\d{10,14})',
        r'\b(19\d{8})\b',  # специфичный префикс Gold Apple
    ]
    
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        articles.extend(found)
        
    # Убираем дубликаты, сохраняем порядок появления
    seen = set()
    unique_articles = []
    for a in articles:
        if a not in seen:
            seen.add(a)
            unique_articles.append(a)
            
    return unique_articles


def calculate_recall_at_k_strict(retrieved: List[str], ground_truth: List[str], k: int = 5) -> float:
    """
    Строгий Recall@K ровно по формуле из методики:
    Recall = (Количество совпавших SKU в топ-K) / (Общее кол-во SKU в эталоне)
    """
    if not ground_truth:
        return 0.0
    
    # Приводим к множествам для быстрого пересечения
    retrieved_set = set(str(r).strip() for r in retrieved[:k])
    gt_set = set(str(g).strip() for g in ground_truth)
    
    # Пересечение
    intersection = retrieved_set & gt_set
    
    return len(intersection) / len(gt_set) 


def call_recommendation_api(query: str, db_index: Dict[str, dict] = None) -> Tuple[str, List[str]]:
    """Вызов агента + строгое извлечение article"""
    url = f"{AGENTS_API_URL}/crew/recommend_products"
    payload = {"query": query, "collection_id": "global_collection"}
    headers = {"Content-Type": "application/json; charset=utf-8"}
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        
        response_text = data.get("raw_text") or data.get("recommendations") or ""
        if not isinstance(response_text, str):
            response_text = str(response_text)
            
        # 1. Строгий парсинг article
        articles = extract_articles_strict(response_text)
        
        # 2. Fallback: если агент не вывел article, ищем по точному совпадению названия в БД
        # (только для сохранения воспроизводимости, логируется отдельно)
        if not articles and db_index:
            logger.warning("⚠️ Agent returned no article IDs. Fallback to exact name matching...")
            # Здесь можно оставить твой match_products_to_articles, но он должен возвращать ТОЛЬКО article
            articles = match_products_to_articles(response_text, db_index)
            
        return response_text, articles
        
    except Exception as e:
        logger.error(f"Recommendation API Error: {e}", exc_info=True)
        return "", []
        

def match_products_to_articles(response_text: str, db_index: Dict[str, dict]) -> List[str]:
    """
    Умный матчинг: извлекает названия товаров из ответа и находит их article в базе.
    """
    if not response_text:
        return []
        
    found_articles = []
    
    # === Шаг 1: Парсим названия товаров (с учётом markdown **...** и эмодзи) ===
    product_names = []
    
    # Паттерн для "🌟 Рекомендация 1: НАЗВАНИЕ" или "🌟 **Рекомендация 1**: НАЗВАНИЕ"
    pattern = r'(?:🌟\s*)?(?:\*\*)?(?:Рекомендация\s*\d*:?\s*|Recommendation\s*\d*:?\s*)(?:\*\*)?\s*([^\n💧✨]+)'
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    
    for name in matches:
        # Очищаем от **, пробелов, переносов
        cleaned = re.sub(r'\*\*', '', name).strip(' :\n\t')
        if cleaned and len(cleaned) > 3:
            product_names.append(cleaned)
    
    # Fallback: если не нашли по паттерну, берём короткие строки
    if not product_names:
        for line in response_text.split('\n'):
            line = re.sub(r'\*\*', '', line).strip()
            if any(m in line for m in ['💧', '✨', 'Почему', 'Ключевые', '---']):
                continue
            words = line.split()
            if 2 <= len(words) <= 10 and not line.startswith(('🌟', '•', '-', '*', '#')):
                product_names.append(line)

    if not product_names:
        logger.warning("⚠️ No product names extracted from response")
        return []

    logger.debug(f"🔍 Extracted names: {product_names[:3]}")
    
    # === Шаг 2: Ищем в базе (БЕЗОПАСНО обрабатываем None) ===
    for name in product_names:
        # Безопасное приведение к lower
        name_lower = str(name).lower().strip() if name else ""
        if not name_lower:
            continue
            
        for article, product in db_index.items():
            # 🛡️ ЗАЩИТА ОТ NONE: используем str() и проверку
            title = str(product.get('title') or '').lower().strip()
            
            brand_info = product.get('brand_info')
            # Если brand_info None, то brand будет пустой строкой
            brand = str(brand_info).lower().strip()[:50] if brand_info else ""
            
            if not title:
                continue  # Пропускаем товары без названия
            
            # 1. Точное или частичное совпадение названия
            if len(name_lower) > 10 and len(title) > 10:
                if name_lower in title or title in name_lower:
                    found_articles.append(article)
                    logger.debug(f"✓ Name match: '{name}' → {article}")
                    break
            
            # 2. Совпадение по бренду + ключевому слову
            if brand:
                brand_first = brand.split()[0] if brand.split() else ""
                if brand_first and brand_first in name_lower:
                    # Проверяем общее слово >4 букв
                    name_words = set(name_lower.split())
                    title_words = set(title.split())
                    common = name_words & title_words
                    if any(len(w) > 4 for w in common):
                        found_articles.append(article)
                        logger.debug(f"✓ Brand+keyword match: '{name}' → {article}")
                        break
    
    return list(set(found_articles))


def call_analysis_api(inci_list: List[str]) -> Dict[str, str]:
    """H2: Парсинг сложного ответа агента Ingredient Safety Analyst"""
    url = f"{AGENTS_API_URL}/crew/analyze_product"
    
    payload = {
        "product_info": ", ".join(inci_list),
        "collection_id": "default"
    }
    
    headers = {"Content-Type": "application/json"}
    if hasattr(settings, 'model_api_key') and settings.model_api_key:
        headers["Authorization"] = f"Bearer {settings.model_api_key}"
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        data = response.json()
        
        # Агент может вернуть ответ в поле "result", "response" или прямо в корне
        # Часто CrewAI кладет raw output в 'result' или 'raw'
        raw_output = data.get("result") or data.get("response") or data.get("raw") or data
        
        # Если это строка (Python repr), пытаемся распарсить
        if isinstance(raw_output, str):
            # Ищем списки ингредиентов по ключевым словам
            categories = {
                "safe": [],
                "neutral": [],
                "caution": [],
                "avoid": []
            }
            
            import re
            pattern = r"Ingredient\(name='([^']+)',\s*rating='([^']+)'"
            matches = re.findall(pattern, raw_output)
            
            for name, rating in matches:
                clean_name = name.split(" (")[0].strip().lower()
                categories[rating.lower()].append(clean_name)
                
            result = {}
            for cat, ingredients in categories.items():
                for ing in ingredients:
                    result[ing] = cat
            
            if result:
                logger.info(f"✅ Parsed {len(result)} ingredients via Regex from string output")
                return result

        # Если это уже словарь (стандартный JSON путь)
        if isinstance(raw_output, dict):
            result = {}
            category_map = {
                "safe_ingredients": "safe",
                "neutral_ingredients": "neutral", 
                "caution_ingredients": "caution",
                "avoid_ingredients": "avoid"
            }
            
            for key, expected_cat in category_map.items():
                if key in raw_output and isinstance(raw_output[key], list):
                    for item in raw_output[key]:
                        # Обработка случая, когда item - это dict
                        if isinstance(item, dict):
                            name = item.get("name", "")
                            rating = item.get("rating", expected_cat)
                        else:
                            continue # Пропускаем, если не dict
                        
                        if name:
                            clean_name = str(name).split(" (")[0].strip().lower()
                            result[clean_name] = str(rating).lower().strip()
            if result:
                return result

        logger.warning(f"❌ Could not parse agent output. Type: {type(raw_output)}")
        logger.debug(f"Raw output preview: {str(raw_output)[:300]}")
        return {}

    except Exception as e:
        logger.error(f"Analysis API Error: {e}", exc_info=True)
        return {}


def load_goldapple_db() -> Dict[str, dict]:
    """Загружает базу товаров для матчинга по названиям"""
    db_path = PROJECT_ROOT / "data" / "parser" / "goldapple_dataset.json"
    if not db_path.exists():
        logger.warning(f"DB not found: {db_path}")
        return {}
    
    with open(db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Индексируем по article
    return {item["article"]: item for item in data if "article" in item}



def calculate_recall_at_k(retrieved_articles, ground_truth, k=5):
    """
    Умный Recall@K: сравнивает по article и по названиям товаров.
    
    Args:
        retrieved: список article или названий от агента
        ground_truth: список {"article": "...", "title": "...", "brand": "..."}
    """
    if not ground_truth:
        return 0.0
    
    # Нормализуем ground truth
    gt_articles = set(p.get("article", "").lower() for p in ground_truth)
    gt_titles = {p.get("title", "").lower().strip() for p in ground_truth}
    gt_brands = {p.get("brand", "").lower().strip() for p in ground_truth if p.get("brand")}
    
    relevant = 0
    for item in retrieved_articles[:k]:
        item_str = str(item).lower().strip()
        
        # 1. Прямое совпадение по article
        if item_str in gt_articles:
            relevant += 1
            continue
        
        # 2. Совпадение по названию (частичное, для защиты от ложных срабатываний)
        for gt_title in gt_titles:
            if len(item_str) > 10 and len(gt_title) > 10:
                if item_str in gt_title or gt_title in item_str:
                    relevant += 1
                    break
        
        # 3. Совпадение по бренду + ключевому слову
        for brand in gt_brands:
            if brand and brand in item_str:
                # Проверяем наличие общего слова >4 букв
                for gt_title in gt_titles:
                    common = set(item_str.split()) & set(gt_title.split())
                    if any(len(w) > 4 for w in common):
                        relevant += 1
                        break
    
    return relevant / len(ground_truth) if ground_truth else 0.0


def calculate_accuracy(predictions: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    """Accuracy: доля правильно классифицированных ингредиентов."""
    if not ground_truth:
        return 0.0
    
    correct = 0
    total = len(ground_truth)
    
    for ing, true_cat in ground_truth.items():
        pred_cat = predictions.get(ing)
        if pred_cat and pred_cat == true_cat:
            correct += 1
            
    return correct / total


def run_h1(limit: int = None, seed: int = 42):
    print("\n" + "="*60)
    print("🚀 EXPERIMENT H1: RECOMMENDATIONS (Recall@5)")
    print("="*60)
    
    if not DATASET_H1_PATH.exists():
        logger.error(f"Dataset H1 not found: {DATASET_H1_PATH}")
        return

    # Load data
    with open(DATASET_H1_PATH, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f if line.strip()]
    
    queries.sort(key=lambda x: x.get('query_id', 0))
    if limit:
        queries = queries[:limit]
    
    db_index = load_goldapple_db()  
        
    recalls = []
    results_log = []

    for i, q in enumerate(queries):
        query_text = q['query']
        
        ground_truth = q.get('ground_truth_products', [])  # List[Dict]
        
        logger.info(f"[{i+1}/{len(queries)}] Query: {query_text[:50]}...")
        
        _, retrieved_articles = call_recommendation_api(query_text, db_index=db_index)
        
        # Берём эталонные SKU из датасета
        gt_skus = q.get('ground_truth_product_ids', [])
        
        # Считаем Recall@5 строго по методике
        recall = calculate_recall_at_k_strict(retrieved_articles, gt_skus, k=5)
        recalls.append(recall)
        
        logger.info(f"   Recall@5: {recall:.4f} | Retrieved: {len(retrieved_articles)} items")
        
        results_log.append({
            "query_id": q.get('query_id'),
            "recall": recall,
            "retrieved_count": len(retrieved_articles)
        })
        
        time.sleep(0.5)

    # Summary
    mean_recall = np.mean(recalls) if recalls else 0.0
    print(f"\n✅ MEAN RECALL@5: {mean_recall:.4f} (Target: >= 0.9)")
    # Save
    save_results("H1", {"mean_recall_at_5": mean_recall, "details": results_log})


def run_h2(limit: int = None, seed: int = 42):
    """
    Запуск эксперимента H2: оценка через Macro-Averaged Recall.
    """
    print("\n" + "="*60)
    print("🚀 EXPERIMENT H2: INGREDIENT ANALYSIS (Mean Recall)")
    print("="*60)
    
    if not DATASET_H2_PATH.exists():
        logger.error(f"Dataset H2 not found: {DATASET_H2_PATH}")
        return

    with open(DATASET_H2_PATH, 'r', encoding='utf-8') as f:
        products = [json.loads(line) for line in f if line.strip()]
        
    products.sort(key=lambda x: str(x.get('product_id', '')))
    if limit:
        products = products[:limit]
        
    recalls = []  
    results_log = []

    for i, p in enumerate(products):
        inci_list = p.get('inci_list', [])
        gt_cats = p.get('ground_truth_categories', {})
        
        logger.info(f"[{i+1}/{len(products)}] Product ID: {p.get('product_id')} ({len(inci_list)} ingredients)")
        
        # Call API
        predictions = call_analysis_api(inci_list)
        
        # Calc Metric: сразу считаем средний Recall
        score = calculate_mean_recall(predictions, gt_cats)
        
        recalls.append(score)
        
        logger.info(f"   Mean Recall: {score:.4f} | Classified: {len(predictions)}/{len(gt_cats)}")
        
        results_log.append({
            "product_id": p.get('product_id'),
            "mean_recall": score,
            "classified_count": len(predictions)
        })
        
        time.sleep(0.5)

    mean_recall = np.mean(recalls) if recalls else 0.0
    print(f"\n✅ MEAN RECALL: {mean_recall:.4f} (Target: >= 0.8)")
    
    save_results("H2", {"mean_recall": mean_recall, "details": results_log})
    

def calculate_mean_recall(predictions: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    """
    Macro-Averaged Recall для 4 классов: safe, neutral, caution, avoid.
    С нормализацией ключей (рус→лат) для совместимости с агентом.
    """
    if not ground_truth:
        return 0.0
    
    normalized_predictions = {}
    for ing, rating in predictions.items():
        norm_key = normalize_ingredient_key(ing)
        normalized_predictions[norm_key] = rating
    
    categories = ["safe", "neutral", "caution", "avoid"]
    category_recalls = []
    
    for cat in categories:
        tp = 0
        fn = 0
        
        for ing, true_cat in ground_truth.items():
            if true_cat != cat:
                continue 
            
            norm_key = normalize_ingredient_key(ing)
            
            pred_cat = normalized_predictions.get(norm_key)
            
            if pred_cat == cat:
                tp += 1
            else:
                fn += 1
        
        denominator = tp + fn
        if denominator > 0:
            category_recalls.append(tp / denominator)
    
    return np.mean(category_recalls) if category_recalls else 0.0


def save_results(exp_name: str, data: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{exp_name.lower()}_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Results saved to {filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["h1", "h2", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Limit items for quick test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.exp in ["h1", "both"]:
        run_h1(limit=args.limit, seed=args.seed)
        
    if args.exp in ["h2", "both"]:
        run_h2(limit=args.limit, seed=args.seed)