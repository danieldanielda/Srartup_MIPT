# node_parser.py
import json
import uuid
from typing import List
from pathlib import Path
from datetime import datetime, timezone
from llama_index.core.schema import TextNode, Document
import logging

logger = logging.getLogger(__name__)

class NodeParser:
    """
    Парсер для JSON-файлов с косметикой.
    """
    async def aload_documents(self, file_paths: List[str]) -> List[Document]:
        documents = []
        current_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        for file_path in file_paths:
            if not file_path.endswith(".json"):
                logger.warning(f"Skipping non-JSON file: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                items = data if isinstance(data, list) else [data]
                file_name = Path(file_path).name
                
                for obj in items:
                    # Формируем текст для эмбеддинга
                    text_parts = []
                    if name := obj.get("name"):
                        text_parts.append(f"Название: {name}")
                    if brand := obj.get("brand"):
                        text_parts.append(f"Бренд: {brand}")
                    if desc := obj.get("description"):
                        text_parts.append(f"Описание: {desc}")
                    if ing := obj.get("ingredients"):
                        text_parts.append(f"Состав: {ing}")
                    if product_type := obj.get("type"):
                        text_parts.append(f"Тип продукта: {product_type}")
                    if skin_type := obj.get("skin_type"):
                        text_parts.append(f"Тип кожи: {skin_type}")
                    
                    full_text = "\n".join(text_parts) or json.dumps(obj, ensure_ascii=False)

                    # ВАЖНО: Сохраняем только простые строковые метаданные для ChromaDB
                    doc = Document(
                        text=full_text,
                        id_=str(uuid.uuid4()),
                        metadata={
                            "source_file": file_path,
                            "file_name": file_name,
                            "creation_date": current_date_utc,
                            "object_type": "cosmetic_product",
                            # Сохраняем только простые поля, не весь объект
                            "product_name": obj.get("name", ""),
                            "product_brand": obj.get("brand", ""),
                            "product_type": obj.get("type", ""),
                            "skin_type": obj.get("skin_type", "")
                        }
                    )
                    documents.append(doc)

            except Exception as e:
                logger.error(f"Failed to load JSON {file_path}: {e}")
                continue

        return documents

    async def _get_user_collection_name(self, user_id: str) -> str:
        """Генерирует имя коллекции для пользователя"""
        return f"user_{user_id}"

    async def acreate_json_nodes(self, user_id: str, documents: List[Document]) -> List[TextNode]:
        """
        Создаёт ноды для JSON с косметикой (без чанкинга).
        ВАЖНО: ChromaDB принимает только простые типы в метаданных (str, int, float, None).
        """
        collection_id = await self._get_user_collection_name(user_id=user_id)
        nodes = []
        
        for doc in documents:
            # Фильтруем метаданные, оставляя только простые типы
            safe_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, type(None))):
                    safe_metadata[key] = value
                else:
                    # Конвертируем сложные типы в строки
                    safe_metadata[key] = str(value)
            
            node = TextNode(
                text=doc.text,
                id_=str(uuid.uuid4()),
                metadata={
                    "parent_id": doc.id_,
                    "collection_id": collection_id,
                    "object_type": "cosmetic_product",
                    **safe_metadata
                },
                # Исключаем сложные метаданные из эмбеддинга и LLM
                excluded_embed_metadata_keys=["parent_id", "collection_id", "source_file"],
                excluded_llm_metadata_keys=["collection_id", "parent_id", "source_file"]
            )
            nodes.append(node)
        
        return nodes