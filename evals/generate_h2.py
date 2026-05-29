# generate_h2_from_inci.py
import json
import random
import os
import re
from pathlib import Path
from typing import Dict, List

random.seed(2026)

# Путь к inci.json
INCI_DB_PATH = "data/inci/inci.json"
OUTPUT_DIR = "evals/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_name(name: str) -> str:
    """
    Приводит название ингредиента к каноническому виду для сравнения:
    - нижний регистр
    - убирает скобки и их содержимое: "Aqua (Water)" -> "aqua"
    - убирает лишние пробелы
    """
    clean = re.split(r'\s*\(', name)[0].strip().lower()
    return clean


def load_inci_db(path: str) -> Dict[str, str]:
    """
    Загружает базу ингредиентов из inci.json (список объектов).
    
    Возвращает словарь: {normalized_name: rating}
    Учитывает оба поля: 'name' и 'inci_name'.
    """
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Если это список объектов (как в твоём примере)
    if isinstance(data, list):
        index = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            
            rating = item.get("rating", "neutral").lower().strip()
            
            # Индексируем по обоим полям имени
            for field in ["name", "inci_name"]:
                if field in item and item[field]:
                    normalized = normalize_name(item[field])
                    if normalized:
                        index[normalized] = rating
        
        print(f"✅ Loaded {len(index)} unique ingredient ratings from {path}")
        return index
    
    # Если вдруг это словарь (старый формат)
    elif isinstance(data, dict):
        return {normalize_name(k): v.get("rating", "neutral").lower().strip() 
                for k, v in data.items()}
    
    return {}


def generate_h2_from_inci(inci_index: Dict[str, str], num_products: int = 100, 
                        min_ingredients: int = 5, max_ingredients: int = 15):
    """
    Генерирует датасет H2, используя inci.json как источник истины.
    """
    if not inci_index:
        print("❌ Cannot generate dataset: inci index is empty")
        return
    
    dataset = []
    all_ingredients = list(inci_index.keys())
    
    print(f"🚀 Generating {num_products} products using {len(inci_index)} ingredients...")
    
    for idx in range(num_products):
        # 1. Случайно выбираем ингредиенты из базы
        num_ing = 100
        selected_normalized = random.sample(all_ingredients, num_ing)
        
        # 2. Формируем ground_truth
        ground_truth = {}
        inci_list_display = []  # Для отправки агенту (красивый вид)
        
        for norm_name in selected_normalized:
            rating = inci_index[norm_name]
            ground_truth[norm_name] = rating
            
            # Для отправки агенту: делаем первую букву заглавной (как в логах)
            display_name = norm_name.capitalize()
            inci_list_display.append(display_name)
        
        dataset.append({
            "product_id": f"sku_test_{idx+1:04d}",
            "product_name": f"Synthetic Product {idx+1}",
            "inci_list": inci_list_display,  # Отправляем агенту в читаемом виде
            "ground_truth_categories": ground_truth,  # Ключи уже normalized (lower)
            "annotator_1": "inci.json_v1",
            "source": "synthetic_from_inci_db",
            "num_ingredients": num_ing
        })
    
    # Сохраняем
    output_path = Path(OUTPUT_DIR) / "eval_dataset_inci.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(dataset)} products to {output_path}")
    
    # Показываем пример
    if dataset:
        sample = dataset[0]
        print(f"\n📊 Sample entry:")
        print(f"   Product ID: {sample['product_id']}")
        print(f"   INCI list (sent to agent): {sample['inci_list'][:5]}...")
        print(f"   Ground Truth (normalized): {dict(list(sample['ground_truth_categories'].items())[:5])}")


if __name__ == "__main__":
    print(f"🔍 Loading INCI database from {INCI_DB_PATH}...")
    inci_index = load_inci_db(INCI_DB_PATH)
    
    if inci_index:
        generate_h2_from_inci(inci_index, num_products=100)
    else:
        print("\n💡 Fallback: please check the path to inci.json")