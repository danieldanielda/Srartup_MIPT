#!/usr/bin/env python3
"""
Генерация датасета H1 на основе реального goldapple_dataset.json
"""

import json
import random
import os
import re
from pathlib import Path
from typing import List, Dict, Set

random.seed(2026)

DB_PATH = "data/parser/goldapple_dataset.json"
OUTPUT_DIR = Path("evals")
OUTPUT_DIR.mkdir(exist_ok=True)


QUERY_TEMPLATES = [
    # Увлажнение / сухая кожа
    {"q": "увлажняющий крем для сухой кожи", "tags": {"сухой", "увлажнение", "крем"}},
    {"q": "интенсивное увлажнение для обезвоженной кожи", "tags": {"сухой", "увлажнение", "обезвоженной"}},
    {"q": "питательный крем для очень сухой кожи", "tags": {"сухой", "питание", "крем"}},
    
    # Жирная / проблемная кожа
    {"q": "средство от прыщей для жирной кожи", "tags": {"жирной", "акне", "очищение"}},
    {"q": "гель для умывания против жирного блеска", "tags": {"жирной", "очищение", "себорегуляция"}},
    {"q": "матирующий крем для комбинированной кожи", "tags": {"жирной", "комбинированной", "матирующий"}},
    
    # Чувствительная кожа
    {"q": "мягкая умывалка для чувствительной кожи", "tags": {"чувствительной", "очищение", "мягкое"}},
    {"q": "крем для кожи склонной к розацеа", "tags": {"чувствительной", "розацеа", "успокаивающий"}},
    {"q": "средство без отдушек для реактивной кожи", "tags": {"чувствительной", "без отдушек", "гипоаллергенный"}},
    
    # Антивозрастной уход
    {"q": "сыворотка от морщин с ретинолом", "tags": {"антивозрастной", "сыворотка", "ретинол"}},
    {"q": "крем с пептидами для упругости кожи", "tags": {"антивозрастной", "пептиды", "лифтинг"}},
    {"q": "ночной антивозрастной уход", "tags": {"антивозрастной", "ночной", "восстановление"}},
    
    # Осветление / пигментация
    {"q": "сыворотка с витамином с от пигментации", "tags": {"осветление", "витамин с", "пигментация"}},
    {"q": "средство от тусклого цвета лица", "tags": {"осветление", "сияние", "тон"}},
    {"q": "крем с ниацинамидом для ровного тона", "tags": {"осветление", "ниацинамид", "тон"}},
    
    # Защита / SPF
    {"q": "солнцезащитный крем для лица спф 50", "tags": {"spf", "защита", "солнце"}},
    {"q": "легкий санскрин под макияж", "tags": {"spf", "легкий", "макияж"}},
    {"q": "spf крем для чувствительной кожи", "tags": {"spf", "чувствительной", "защита"}},
    
    # Очищение
    {"q": "гель для умывания без сульфатов", "tags": {"очищение", "без sls", "мягкое"}},
    {"q": "мицеллярная вода для снятия макияжа", "tags": {"очищение", "демакияж", "мицеллярная"}},
    {"q": "пенка для умывания для сухой кожи", "tags": {"очищение", "сухой", "пенка"}},
    
    # Сыворотки / активы
    {"q": "сыворотка с гиалуроновой кислотой", "tags": {"сыворотка", "гиалуроновая", "увлажнение"}},
    {"q": "концентрат с кислотами для обновления кожи", "tags": {"сыворотка", "кислоты", "обновление"}},
]

def generate_query_variations(base_query: str, tags: Set[str], num_variations: int = 5) -> List[Dict]:
    """
    Генерирует вариации одного запроса с сохранением семантики.
    """
    # Синонимы для рандомизации
    synonyms = {
        "крем": ["крем", "эмульсия", "флюид", "бальзам"],
        "сыворотка": ["сыворотка", "концентрат", "эликсир", "ампула"],
        "гель": ["гель", "пенка", "мусс", "желе"],
        "для": ["для", "при", "от", "против", "при"],
        "кожи": ["кожи", "лица", "эпидермиса"],
        "увлажняющий": ["увлажняющий", "гидратирующий", "насыщающий влагой"],
        "мягкий": ["мягкий", "деликатный", "бережный", "щадящий"],
    }
    
    variations = []
    for i in range(num_variations):
        varied_query = base_query
        for word, syns in synonyms.items():
            if word in varied_query and random.random() < 0.3:
                varied_query = varied_query.replace(word, random.choice(syns), 1)
        
        variations.append({
            "q": varied_query,
            "tags": tags.copy()  # Теги те же
        })
    
    return variations

KEYWORD_MAP = {
    "сухой": ["сухой", "сухость", "dry", "обезвоженной"],
    "жирной": ["жирной", "жирность", "oily", "себум", "блеск"],
    "чувствительной": ["чувствительной", "sensitive", "раздраженной", "розацеа", "купероз"],
    "нормальной": ["нормальной", "normal", "комбинированной"],
    "очищение": ["очищение", "cleansing", "умывание", "демакияж", "гель", "пенка"],
    "увлажнение": ["увлажнение", "hydration", "влажность", "гидратация"],
    "антивозрастной": ["антивозрастной", "anti-age", "морщин", "омоложение", "лифтинг"],
    "spf": ["spf", "солнцезащитный", "uv", "защита от солнца"],
    "акне": ["акне", "прыщи", "acne", "воспаления", "себорегуляция"],
    "без sls": ["без сульфатов", "без sls", "без sles", "мягкое очищение"],
    "витамин с": ["витамин с", "ascorbic", "осветление", "пигментация", "brightening"],
}


def extract_tags_from_product(product: Dict) -> Set[str]:
    """
    Извлекает теги из полей товара: description, characteristics, ingredients.
    Безопасно обрабатывает None значения.
    """
    tags = set()
    
    # Helper для безопасного получения строки
    def safe_str(val):
        return str(val).lower() if val is not None else ""
    
    text_parts = [
        safe_str(product.get("title")),
        safe_str(product.get("description")),
        safe_str(product.get("ingredients")), 
        safe_str(product.get("brand_info")),
        safe_str(product.get("characteristics")),
    ]
    
    text = " ".join(text_parts)
    
    # Ищем ключевые слова
    for tag, keywords in KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            tags.add(tag)
    
    # Добавляем тип продукта из characteristics
    char = product.get("characteristics", {})
    if isinstance(char, dict) and char.get("тип продукта"):
        tags.add(str(char["тип продукта"]).lower())
    
    # Добавляем бренд (первое слово из brand_info)
    brand_info = product.get("brand_info", "")
    if brand_info:
        # Берем часть до тире или первое слово
        brand_part = str(brand_info).split("–")[0].strip()
        if brand_part:
            brand = brand_part.split()[0].lower()
            tags.add(f"brand:{brand}")
    
    return tags


def load_real_db(path: Path) -> List[Dict]:
    """Загружает реальную базу товаров."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_h1_from_real_db(products: List[Dict], target_queries: int = 100) -> List[Dict]:
    """
    Генерирует ровно target_queries запросов на основе реальной базы.
    """
    print(f"🔍 Indexing {len(products)} products...")
    
    # Индексируем товары
    tagged_products = []
    for p in products:
        tags = extract_tags_from_product(p)
        if tags:
            tagged_products.append({
                "article": p["article"],
                "title": p["title"],
                "brand": p.get("brand_info", "").split("–")[0].strip() if p.get("brand_info") else "",
                "tags": tags,
            })
    
    print(f"✅ Indexed {len(tagged_products)} products with tags")
    
    dataset = []
    query_id = 1
    
    all_query_templates = []
    for tmpl in QUERY_TEMPLATES:
        variations = generate_query_variations(tmpl["q"], tmpl["tags"], num_variations=6)
        all_query_templates.extend(variations)
    
    # Перемешиваем и берём нужное количество
    random.shuffle(all_query_templates)
    
    for tmpl in all_query_templates[:target_queries]:
        query_text = tmpl["q"]
        target_tags = tmpl["tags"]
        
        # Находим релевантные товары
        relevant = [p for p in tagged_products if target_tags & p["tags"]]
        
        # Если мало релевантных — берём случайные с тем же типом
        if len(relevant) < 3:
            product_type_words = query_text.split()[-2:]  # последние 2 слова
            others = [
                p for p in tagged_products if p not in relevant
                if any(w in str(p["tags"]) for w in product_type_words)
            ]
            relevant += random.sample(others, min(3, len(others)))
        
        # Ground Truth: топ-5
        ground_truth = random.sample(relevant, min(5, len(relevant)))
        
        record = {
            "query_id": f"nl_{query_id:03d}",
            "query": query_text,
            "ground_truth_articles": [p["article"] for p in ground_truth],
            "ground_truth_products": [
                {
                    "article": p["article"],
                    "title": p["title"],
                    "brand": p["brand"],
                    "matched_tags": list(target_tags & p["tags"])
                }
                for p in ground_truth
            ],
            "annotator_1": "auto_generated",
            "source": "real_goldapple_db",
            "metadata_tags": list(target_tags)
        }
        
        dataset.append(record)
        query_id += 1
        
        if query_id % 20 == 0:
            print(f"   ✓ Generated {query_id}/{target_queries} queries...")
    
    print(f"✅ Generated {len(dataset)} queries total")
    return dataset

def main():
    
    print(f"🚀 Generating H1 dataset: 100 queries from real database...")
    
    products = load_real_db(DB_PATH)
    
    # Генерируем ровно 100 запросов
    dataset = generate_h1_from_real_db(products, target_queries=100)
    
    # Сохраняем
    output_path = OUTPUT_DIR / "eval_dataset_nl_queries.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved 100 queries to {output_path}")

if __name__ == "__main__":
    main()