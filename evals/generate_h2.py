#!/usr/bin/env python3
"""
Пересборка датасета H2 с эталоном на латинице (INCI).
Берёт реальные составы из goldapple_dataset.json (поле ingredients — уже INCI).
Разметка — через LLM-as-Judge (EWG/CIR/EU).
"""

import json
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from config import EvalSettings

random.seed(2026)

GOLDAPPLE_DB_PATH = Path("data/parser/goldapple_dataset.json")
OUTPUT_DIR = Path("evals")
OUTPUT_DIR.mkdir(exist_ok=True)

settings = EvalSettings()
llm_client = OpenAI(
    api_key=settings.model_api_key,
    base_url=settings.model_api,
    timeout=60,
    max_retries=3
)

INGREDIENT_CACHE_PATH = OUTPUT_DIR / "inci_judge_cache_inci.json"

def normalize_inci_name(name: str) -> str:
    """
    Приводит INCI-название к каноническому виду:
    - нижний регистр
    - убирает синонимы через / : "Aqua/water/eau" → "aqua"
    - убирает скобки и их содержимое
    - убирает лишние пробелы, %, спецсимволы
    """
    # Берём первую часть до / (основное INCI-имя)
    clean = name.split('/')[0].strip().lower()
    # Убираем скобки и их содержимое
    clean = re.split(r'\s*\(', clean)[0].strip()
    # Убираем проценты, лишние символы
    clean = re.sub(r'[^a-z0-9\s\-]', '', clean)
    # Нормализуем пробелы
    clean = ' '.join(clean.split())
    return clean


def load_ingredient_cache() -> Dict[str, str]:
    if INGREDIENT_CACHE_PATH.exists():
        with open(INGREDIENT_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_ingredient_cache(cache: Dict[str, str]):
    with open(INGREDIENT_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def classify_ingredient_with_llm(ingredient_name: str, cache: Dict[str, str]) -> Optional[str]:
    """Классифицирует ингредиент через LLM-as-Judge"""
    normalized = normalize_inci_name(ingredient_name)
    
    if normalized in cache:
        return cache[normalized]
    
    prompt = f"""Ты — эксперт-косметолог и регуляторный аналитик.
Оцени уровень безопасности косметического ингредиента на основе баз:
- EU Cosmetics Regulation 1223/2009 (Annex II-VI)
- EWG Skin Deep Database
- Cosmetic Ingredient Review (CIR)

Ингредиент (INCI): "{ingredient_name}"

Верни ТОЛЬКО одно слово из четырёх:
- safe: безопасен при обычном использовании
- neutral: нейтрален, нет данных о вреде или пользе
- caution: требует осторожности (потенциальный аллерген, раздражитель)
- avoid: рекомендуется избегать (токсичен, запрещён, высокий риск)

Ответ (только одно слово):"""
    
    try:
        response = llm_client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "Отвечай строго по инструкции. Только одно слово: safe, neutral, caution или avoid."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        verdict = response.choices[0].message.content.strip().lower()
        
        if verdict in ["safe", "neutral", "caution", "avoid"]:
            cache[normalized] = verdict
            return verdict
        else:
            for cat in ["safe", "neutral", "caution", "avoid"]:
                if cat in verdict:
                    cache[normalized] = cat
                    return cat
            return None
            
    except Exception as e:
        print(f"⚠️  Error classifying '{ingredient_name}': {e}")
        return None


def parse_ingredients_from_inci_field(ingredients_field: str) -> List[str]:
    """
    Парсит поле ingredients из goldapple_dataset.json (формат: "Aqua/water/eau, Propanediol, ...")
    Возвращает список чистых INCI-названий (латиница).
    """
    if not ingredients_field:
        return []
    
    ingredients = []
    for item in ingredients_field.split(','):
        # Берём основное имя (до /)
        clean = item.split('/')[0].strip()
        # Убираем скобки и их содержимое
        clean = re.split(r'\s*\(', clean)[0].strip()
        # Убираем проценты, лишние символы в конце
        clean = re.sub(r'\s*[<\(].*$', '', clean).strip()
        
        # Фильтруем: должно начинаться с латинской буквы и быть длиннее 2 символов
        if clean and len(clean) > 2 and re.match(r'^[a-zA-Z]', clean):
            ingredients.append(clean)
    
    return ingredients


def load_goldapple_products() -> List[Dict]:
    if not GOLDAPPLE_DB_PATH.exists():
        print(f"❌ Database not found: {GOLDAPPLE_DB_PATH}")
        return []
    
    with open(GOLDAPPLE_DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def rebuild_h2_inci_dataset(num_products: int = 100, min_ingredients: int = 5, max_ingredients: int = 50):
    """Пересобирает датасет H2 с эталоном на латинице (INCI)"""
    print(f"🚀 Rebuilding H2 dataset with INCI (Latin) ground truth...")
    
    products = load_goldapple_products()
    cache = load_ingredient_cache()
    
    if not products:
        print("❌ No products found")
        return
    
    # Фильтруем продукты с валидным INCI-составом (латиница)
    valid_products = [
        p for p in products 
        if p.get('ingredients') and p.get('article') and re.search(r'[a-zA-Z]', p.get('ingredients', ''))
    ]
    
    print(f"📦 Found {len(valid_products)} products with INCI (Latin) ingredients")
    
    random.shuffle(valid_products)
    
    dataset = []
    total_classifications = 0
    
    for product in valid_products:
        if len(dataset) >= num_products:
            break
            
        article = product['article']
        title = product.get('title', 'Unknown')
        ingredients_field = product.get('ingredients', '')
        
        # Парсим INCI-состав (латиница)
        inci_list = parse_ingredients_from_inci_field(ingredients_field)
        
        if not (min_ingredients <= len(inci_list) <= max_ingredients):
            continue
        
        ground_truth = {}
        current_idx = len(dataset) + 1
        print(f"[{current_idx}/{num_products}] Product: {title[:50]}... ({len(inci_list)} INCI ingredients)")
        
        for ing in inci_list:
            normalized = normalize_inci_name(ing)
            if normalized in ground_truth:
                continue  # дубликаты
            
            rating = classify_ingredient_with_llm(ing, cache)
            if rating:
                ground_truth[normalized] = rating
                total_classifications += 1
        
        # Сохраняем кэш периодически
        if current_idx % 10 == 0:
            save_ingredient_cache(cache)
            time.sleep(1)  # rate limiting
        
        if ground_truth:
            dataset.append({
                "product_id": article,
                "product_name": title,
                "inci_list": inci_list,  # Список на латинице для отправки агенту
                "ground_truth_categories": ground_truth,  # Ключи тоже на латинице!
                "annotator_1": "LLM-as-Judge (EWG/CIR/EU)",
                "source": "goldapple_real_products_inci",
                "num_ingredients": len(ground_truth)
            })
        
        time.sleep(0.2)
    
    # Финальное сохранение кэша
    save_ingredient_cache(cache)
    
    # Сохраняем датасет
    output_path = OUTPUT_DIR / "eval_dataset_inci.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved {len(dataset)} products to {output_path}")
    print(f"📊 Total ingredient classifications: {total_classifications}")
    
    # Статистика по классам
    if dataset:
        class_counts = {"safe": 0, "neutral": 0, "caution": 0, "avoid": 0}
        for record in dataset:
            for cat in record["ground_truth_categories"].values():
                class_counts[cat] = class_counts.get(cat, 0) + 1
        
        print(f"📈 Class distribution:")
        for cat, count in class_counts.items():
            pct = count / total_classifications * 100 if total_classifications > 0 else 0
            print(f"   {cat}: {count} ({pct:.1f}%)")
    
    # Пример записи
    if dataset:
        sample = dataset[0]
        print(f"\n📋 Sample entry (INCI/Latin):")
        print(f"   Product: {sample['product_name']}")
        print(f"   Article: {sample['product_id']}")
        print(f"   INCI list: {sample['inci_list'][:5]}...")
        print(f"   Ground Truth (first 5): {dict(list(sample['ground_truth_categories'].items())[:5])}")


if __name__ == "__main__":
    rebuild_h2_inci_dataset(num_products=100, min_ingredients=5, max_ingredients=50)