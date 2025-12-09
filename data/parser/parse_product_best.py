from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import random
import re
import time

# Парсим категории
def parse_product_characteristics_from_soup(soup):
    characteristics = {}
    characteristics_section = soup.find('dl', class_=lambda x: x and 'sPUOq' in x)
    if characteristics_section:
        char_blocks = characteristics_section.find_all('div', class_=lambda x: x and 'uA+oO' in x)
        for block in char_blocks:
            name_el = block.find('dt', class_=lambda x: x and 'kEHrl' in x)
            value_el = block.find('dt', class_=lambda x: x and '_4Gkwv' in x)
            if name_el and value_el:
                name = name_el.get_text(strip=True)
                value = value_el.get_text(strip=True)
                characteristics[name] = value
    return characteristics

def extract_category_from_breadcrumbs(soup):
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
        page.wait_for_load_state('networkidle')
        html_content = page.content()
        browser.close()

    soup = BeautifulSoup(html_content, 'html.parser')
    product_data = {}

    # 1. Название
    title_el = soup.find('div', class_=lambda x: x and 'fgoWE' in x)
    product_data['title'] = title_el.get_text(strip=True) if title_el else None

    # 2. Артикул
    article_el = soup.find('div', class_=lambda x: x and '_09clY' in x)
    article_text = article_el.get_text(strip=True) if article_el else ''
    article_match = re.search(r'\d+', article_text)
    product_data['article'] = article_match.group() if article_match else None

    # 3–7. Вкладки
    tabs = {
        'Описание': 'description',
        'Применение': 'application',
        'Состав': 'ingredients',
        'Бренд': 'brand_info',
        'Дополнительная информация': 'additional_info'
    }
    for tab_name, key in tabs.items():
        tab_block = soup.find('div', attrs={'text': tab_name})
        if tab_block:
            text_el = tab_block.find('div', class_=lambda x: x and '_1ujdd' in x)
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

def parse_category_products_infinite_scroll(category_url, max_products=None):
    """
    Улучшенная функция сбора ссылок с бесконечной прокруткой
    """
    print(f"🔄 Загружаем категорию: {category_url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled']
        )
        
        # Убираем автоматизационные признаки
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()
        page.set_default_timeout(30000)
        
        # Переходим на страницу
        try:
            page.goto(category_url, wait_until='networkidle')
            print("✅ Страница загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки страницы: {e}")
            browser.close()
            return []

        product_urls = set()
        scroll_attempts = 0
        max_scroll_attempts = 100
        no_new_count = 0
        max_no_new = 8
        
        last_count = 0
        
        print("🎯 Начинаем сбор товаров...")
        
        while (scroll_attempts < max_scroll_attempts and 
            no_new_count < max_no_new):
            
            scroll_attempts += 1
            print(f"📜 Скролл {scroll_attempts}...")
            
            # Прокрутка с рандомизацией
            scroll_height = page.evaluate("document.body.scrollHeight")
            current_position = page.evaluate("window.pageYOffset")
            
            # Прокручиваем разными способами
            scroll_methods = [
                f"window.scrollTo(0, {scroll_height})",
                "window.scrollBy(0, 1500)",
                f"window.scrollTo(0, {current_position + 1200})"
            ]
            
            for method in scroll_methods:
                try:
                    page.evaluate(method)
                    time.sleep(random.uniform(1, 2))
                except:
                    pass
            
            # Ждем загрузки
            time.sleep(random.uniform(2, 4))
            
            current_urls = set()
            
            all_links = page.query_selector_all('a')
            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href and re.match(r'^/\d{5,}-[a-zA-Z0-9-]+', href):
                        full_url = f"https://goldapple.ru{href}"
                        current_urls.add(full_url)
                except:
                    continue
            
            # Альтернативный способ: ищем по data-атрибутам
            product_elements = page.query_selector_all('[data-scroll-id]')
            for element in product_elements:
                try:
                    link_element = element.query_selector('a')
                    if link_element:
                        href = link_element.get_attribute('href')
                        if href and re.match(r'^/\d{5,}-[a-zA-Z0-9-]+', href):
                            full_url = f"https://goldapple.ru{href}"
                            current_urls.add(full_url)
                except:
                    continue
            
            # Проверяем новые товары
            new_urls = current_urls - product_urls
            current_count = len(current_urls)
            
            if new_urls:
                product_urls.update(new_urls)
                no_new_count = 0
                print(f"✅ +{len(new_urls)} новых товаров (всего: {len(product_urls)})")
            else:
                no_new_count += 1
                print(f"➖ Новых нет ({no_new_count}/{max_no_new})")
            
            # Проверяем лимит
            if max_products and len(product_urls) >= max_products:
                print(f"🎯 Достигнут лимит {max_products} товаров")
                break
            
            # Проверяем конец страницы
            new_scroll_height = page.evaluate("document.body.scrollHeight")
            if new_scroll_height == scroll_height:
                no_new_count += 1
            else:
                no_new_count = max(0, no_new_count - 0.5)
            
            # Случайная пауза
            time.sleep(random.uniform(1, 3))
        
        print(f"🏁 Завершено. Всего собрано: {len(product_urls)} товаров")
        
        # Финальный сбор всех возможных ссылок
        print("🔍 Финальный сбор ссылок...")
        final_links = page.query_selector_all('a')
        for link in final_links:
            try:
                href = link.get_attribute('href')
                if href and re.match(r'^/\d{5,}-[a-zA-Z0-9-]+', href):
                    full_url = f"https://goldapple.ru{href}"
                    product_urls.add(full_url)
            except:
                continue
        browser.close()

    result = list(product_urls)
    if max_products:
        result = result[:max_products]
    return result

def main():
    """Улучшенная основная функция"""
    CATEGORY_URL = "https://goldapple.ru/uhod/uhod-za-licom/ochischenie-i-demakijazh"
    
    print("🚀 Запуск парсера Goldapple...")
    
    # Сбор ссылок
    print("🔍 Этап 1: Сбор ссылок на товары")
    product_urls = parse_category_products_infinite_scroll(
        CATEGORY_URL, 
        max_products=200
    )
    
    if not product_urls:
        print("❌ Не удалось собрать ссылки на товары")
        return
    
    print(f"📦 Найдено {len(product_urls)} товаров")
    
    # Парсинг товаров
    print("\n🔍 Этап 2: Парсинг информации о товарах")
    all_products = []
    
    for i, url in enumerate(product_urls, 1):
        print(f"[{i}/{len(product_urls)}] Обрабатываем: {url}")
        
        try:
            product_info = parse_full_product_info(url)
            all_products.append(product_info)
            
            if 'error' in product_info:
                print(f"   ❌ Ошибка: {product_info['error']}")
            else:
                print(f"   ✅ Успешно: {product_info.get('title', 'No title')}")
                
        except Exception as e:
            print(f"   💥 Критическая ошибка: {e}")
            all_products.append({'url': url, 'error': str(e)})
        
        # Пауза с рандомизацией
        time.sleep(random.uniform(1, 3))
    
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