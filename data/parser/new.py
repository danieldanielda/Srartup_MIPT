from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Парсим категории
def parse_product_characteristics_from_soup(soup):
    characteristics = {}
    # Новый класс dl: VJTw-
    characteristics_section = soup.find('dl', class_=lambda x: x and 'VJTw-' in x)
    if characteristics_section:
        # Обёртка для пар: Din91
        char_blocks = characteristics_section.find_all('div', class_=lambda x: x and 'Din91' in x)
        for block in char_blocks:
            # Ключ: _0wFa-
            name_el = block.find('dt', class_=lambda x: x and '_0wFa-' in x)
            # Значение: SPxyR
            value_el = block.find('dt', class_=lambda x: x and 'SPxyR' in x)
            if name_el and value_el:
                name = name_el.get_text(strip=True)
                value = value_el.get_text(strip=True)
                characteristics[name] = value
    return characteristics

def extract_category_from_breadcrumbs(soup):
    # GoldApple использует JSON-LD, а не <ol itemtype=...>
    # Но оставляем как есть — вдруг появится
    breadcrumb_list = soup.find('ol', {'itemtype': 'https://schema.org/BreadcrumbList'})
    if breadcrumb_list:
        items = breadcrumb_list.find_all('li', {'itemtype': 'https://schema.org/ListItem'})
        if items:
            last = items[-1]
            name_span = last.find('span', itemprop='name')
            if name_span:
                return name_span.get_text(strip=True)
    return None

def parse_full_product_info(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url.strip())
        # Добавим ожидание появления контента — иначе будет "включите JS"
        page.wait_for_selector('div[data-test-id="content"]', timeout=10000)
        html_content = page.content()
        browser.close()

    soup = BeautifulSoup(html_content, 'html.parser')
    product_data = {}

    # 1. Название — теперь в div с классом _2PnS+
    title_el = soup.find('div', class_=lambda x: x and '_2PnS+' in x)
    product_data['title'] = title_el.get_text(strip=True) if title_el else None

    # 2. Артикул — в div с классом rQ0IS
    article_el = soup.find('div', class_=lambda x: x and 'rQ0IS' in x)
    article_text = article_el.get_text(strip=True) if article_el else ''
    article_match = re.search(r'\d+', article_text)
    product_data['article'] = article_match.group() if article_match else None

    # 3–7. Вкладки — ищем по тексту внутри div с атрибутом text (он есть в HTML!)
    tabs = {
        'Описание': 'description',
        'Применение': 'application',
        'Состав': 'ingredients',
        'Бренд': 'brand_info',
        'Дополнительная информация': 'additional_info'
    }
    for tab_name, key in tabs.items():
        # Ищем div с атрибутом text="..."
        tab_block = soup.find('div', attrs={'text': tab_name})
        if tab_block:
            # Текст в div.IEOAL
            text_el = tab_block.find('div', class_=lambda x: x and 'IEOAL' in x)
            if text_el:
                product_data[key] = text_el.get_text(separator=' ', strip=True)
            else:
                product_data[key] = None
        else:
            product_data[key] = None

    # 8. Характеристики
    product_data['characteristics'] = parse_product_characteristics_from_soup(soup)

    # 9. Категория (из breadcrumbs)
    product_data['category'] = extract_category_from_breadcrumbs(soup)

    return product_data

def parse_category_products_with_pagination(category_url, max_products=None):
    """
    Функция сбора ссылок с учетом пагинации
    """
    print(f"🔄 Загружаем категорию: {category_url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled']
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()
        page.set_default_timeout(45000)
        
        product_urls = set()
        page_num = 1
        
        print("🎯 Начинаем обход страниц пагинации...")
        
        # Сначала обрабатываем первую страницу (без параметра p)
        print(f"📄 Страница 1: {category_url}")
        
        try:
            page.goto(category_url, wait_until='domcontentloaded')
            time.sleep(3)
            
            # Собираем товары с первой страницы
            current_page_urls = set()
            all_links = page.query_selector_all('a[href]')
            
            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href and re.match(r'^/\d{5,}-[a-zA-Z0-9-]+', href):
                        full_url = f"https://goldapple.ru{href}"
                        current_page_urls.add(full_url)
                except:
                    continue
            
            if current_page_urls:
                product_urls.update(current_page_urls)
                print(f"✅ Страница 1: +{len(current_page_urls)} товаров (всего: {len(product_urls)})")
            else:
                print("❌ На первой странице не найдено товаров")
                return []
            
        except Exception as e:
            print(f"❌ Ошибка на первой странице: {e}")
            return []
        
        # Теперь обрабатываем остальные страницы (начиная со второй)
        page_num = 2
        has_next_page = True
        
        while has_next_page:
            # Формируем URL с пагинацией
            if '?' in category_url:
                current_url = f"{category_url}&p={page_num}"
            else:
                current_url = f"{category_url}?p={page_num}"
            
            print(f"📄 Страница {page_num}: {current_url}")
            
            try:
                page.goto(current_url, wait_until='domcontentloaded')
                time.sleep(3)
                
                # Проверяем, есть ли товары на странице
                current_page_urls = set()
                all_links = page.query_selector_all('a[href]')
                
                for link in all_links:
                    try:
                        href = link.get_attribute('href')
                        if href and re.match(r'^/\d{5,}-[a-zA-Z0-9-]+', href):
                            full_url = f"https://goldapple.ru{href}"
                            current_page_urls.add(full_url)
                    except:
                        continue
                
                # Если на странице нет товаров - значит это последняя страница
                if not current_page_urls:
                    print(f"➖ Страница {page_num} пустая - завершаем")
                    has_next_page = False
                    break
                
                # Добавляем найденные товары
                new_urls = current_page_urls - product_urls
                if new_urls:
                    product_urls.update(new_urls)
                    print(f"✅ Страница {page_num}: +{len(new_urls)} товаров (всего: {len(product_urls)})")
                else:
                    print(f"⚠️  Страница {page_num}: повторяющиеся товары")
                    # Если несколько страниц подряд повторяются - выходим
                    if page_num > 10:  # защита от бесконечного цикла
                        has_next_page = False
                        break
                
                # Проверяем лимит
                if max_products and len(product_urls) >= max_products:
                    print(f"🎯 Достигнут лимит {max_products} товаров")
                    break
                
                page_num += 1
                
                # Защита от бесконечного цикла
                if page_num > 1500:  # максимальное количество страниц
                    print("⚠️  Достигнут лимит в 50 страниц")
                    break
                
                # Пауза между страницами
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                print(f"❌ Ошибка на странице {page_num}: {e}")
                # Если ошибка на второй странице - возможно, пагинации нет
                if page_num == 2:
                    print("ℹ️  Возможно, у категории только одна страница")
                    break
                else:
                    # Пробуем следующую страницу
                    page_num += 1
                    if page_num > 5:  # если несколько ошибок подряд - выходим
                        print("💥 Слишком много ошибок, завершаем")
                        break
        
        browser.close()
    
    result = list(product_urls)
    if max_products:
        result = result[:max_products]
    
    print(f"📦 Собрано всего: {len(result)} товаров с {page_num-1} страниц")
    return result

def parse_single_product(url):
    """Функция для парсинга одного товара — должна быть сериализуемой (top-level)."""
    try:
        return parse_full_product_info(url)
    except Exception as e:
        return {'url': url, 'error': str(e)}
    
def main():
    """Улучшенная основная функция"""
    CATEGORY_URL = "https://goldapple.ru/uhod/uhod-za-licom/antivozrastnoj-uhod-za-licom"
    
    print("🚀 Запуск парсера Goldapple...")
    
    # Сбор ссылок
    print("🔍 Этап 1: Сбор ссылок на товары")
    product_urls = parse_category_products_with_pagination(
        CATEGORY_URL, 
        max_products=None  # Все товары
    )
    
    if not product_urls:
        print("❌ Не удалось собрать ссылки на товары")
        return
    
    print(f"📦 Найдено {len(product_urls)} товаров")
    
    # Парсинг товаров
    print("\n🔍 Этап 2: Парсинг информации о товарах")
    # Определяем оптимальное число процессов
    MAX_WORKERS = min(8, os.cpu_count() or 4)  # не более 8, чтобы не перегружать сеть/сервер

    print(f"\n🔍 Этап 2: Парсинг {len(product_urls)} товаров с использованием {MAX_WORKERS} процессов...")

    all_products = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Отправляем задачи
        future_to_url = {executor.submit(parse_single_product, url): url for url in product_urls}
        
        for i, future in enumerate(as_completed(future_to_url), 1):
            url = future_to_url[future]
            try:
                result = future.result()
                all_products.append(result)
                title = result.get('title', 'No title') if 'error' not in result else 'ERROR'
                print(f"[{i}/{len(product_urls)}] ✅ {title}")
            except Exception as e:
                print(f"[{i}/{len(product_urls)}] 💥 Ошибка при получении результата для {url}: {e}")
                all_products.append({'url': url, 'error': str(e)})
        
        # Пауза с рандомизацией
        time.sleep(random.uniform(0.5, 1.5))
    
    # Сохранение результатов
    print("\n💾 Сохранение результатов...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'goldapple_products_{timestamp}.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_products, f, ensure_ascii=False, indent=4)
    
    # Статистика
    successful = len([p for p in all_products if 'error' not in p])
    errors = len(all_products) - successful
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"✅ Успешно: {successful}")
    print(f"❌ С ошибками: {errors}")
    print(f"📁 Файл: {filename}")
    
    if successful > 0:
        print(f"🎯 Примеры товаров:")
        for product in all_products[:3]:
            if 'error' not in product:
                print(f"   • {product.get('title', 'No title')}")

if __name__ == "__main__":
    main()