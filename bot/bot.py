import os
import tempfile
import cv2
import numpy as np
from pyzbar import pyzbar
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, ContentType
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardMarkup
from aiogram.enums import ParseMode

BOT_TOKEN = ""

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

def decode_barcode_with_cv2(image_path: str) -> str | None:
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    attempts = [
        gray,
        cv2.equalizeHist(gray),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]

    for img in attempts:
        barcodes = pyzbar.decode(img)
        if barcodes:
            return barcodes[0].data.decode("utf-8")

    for scale in [1.2, 1.5, 2.0]:
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        barcodes = pyzbar.decode(resized)
        if barcodes:
            return barcodes[0].data.decode("utf-8")

    return None

def get_ingredients_by_barcode(barcode: str) -> str | None:

    try:
        return "Это косметика" + barcode
    except Exception as e:
        print(f"[ERROR] API request failed: {e}")
    return None

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "📸 Пришлите фото штрихкода косметики — я постараюсь найти её состав!",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[[{"text": "ℹ️ Как снимать штрихкод"}]],
            resize_keyboard=True,
            one_time_keyboard=False
        )
    )

@router.message(F.text == "ℹ️ Как снимать штрихкод")
async def how_to_scan(message: Message):
    await message.answer(
        "📌 Советы:\n"
        "• Убедитесь, что штрихкод чёткий и не размыт\n"
        "• Хорошее освещение — обязательно\n"
        "• Держите камеру параллельно штрихкоду\n"
        "• Избегайте бликов и теней"
    )

@router.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message):
    # Получаем фото самого высокого качества
    photo = message.photo[-1]
    
    # Создаём временный файл
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        await bot.download(photo, destination=tmp_path)

        barcode = decode_barcode_with_cv2(tmp_path)

        if not barcode:
            await message.reply(
                "❌ Не удалось распознать штрихкод.\n\n"
                "Попробуйте сделать фото чётче или в другом ракурсе."
            )
            return

        # Запрашиваем состав
        ingredients = get_ingredients_by_barcode(barcode)

        if ingredients:
            await message.reply(
                f"✅ Найден штрихкод: `{barcode}`\n\n{ingredients}",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await message.reply(
                f"🔍 Штрихкод распознан: `{barcode}`\n\n"
                "Но информация о продукте не найдена в открытой базе (Open Beauty Facts).\n"
                "Попробуйте другой продукт или проверьте актуальность данных на https://world.openbeautyfacts.org"
            )

    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# === ЗАПУСК ===
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())