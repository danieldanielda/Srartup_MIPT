from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import random
import re
import time

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

if __name__ == "__main__":
    x = parse_full_product_info("https://goldapple.ru/6030400045-skin-naturals")
    print(x)