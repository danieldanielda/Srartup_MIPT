import os
from typing import List
from llama_index.core.schema import TextNode
from node_parser import NodeParser
import logging

logger = logging.getLogger(__name__)
class DocumentProcessor:
    def __init__(self) -> None:
        self.parser = NodeParser()

    async def aprocess_directory(self, user_id: str, directory_path: str, **kwargs) -> List[TextNode]:
        """Обрабатывает директорию с JSON файлами косметики"""
        file_paths = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".json")
        ]
        
        if not file_paths:
            logger.warning(f"No JSON files found in {directory_path}")
            return []
            
        logger.info(f"Processing {len(file_paths)} JSON files from {directory_path}")
        
        # Загружаем документы
        documents = await self.parser.aload_documents(file_paths)
        
        # Создаём ноды (без чанкинга для JSON)
        nodes = await self.parser.acreate_json_nodes(user_id=user_id, documents=documents)
        
        logger.info(f"Created {len(nodes)} cosmetic product nodes from {len(file_paths)} files")
        return nodes

    async def process_single_file(self, user_id: str, file_path: str, **kwargs) -> List[TextNode]:
        """Обрабатывает один JSON файл с косметикой"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.endswith(".json"):
            raise ValueError("Only JSON files are supported for cosmetic data")
        
        try:
            documents = await self.parser.aload_documents([file_path])
            nodes = await self.parser.acreate_json_nodes(user_id=user_id, documents=documents)
            logger.info(f"Successfully processed {file_path} into {len(nodes)} cosmetic product nodes")
            return nodes
            
        except Exception as e:
            logger.error(f"Error processing cosmetic file {file_path}: {e}")
            raise