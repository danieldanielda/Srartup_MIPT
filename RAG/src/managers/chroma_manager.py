import uuid
import chromadb
from typing import List
from chromadb.config import Settings as Chroma_Settings
import logging

logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Manager for working with ChromaDB in docker container.
    Provides a convenient interface for managing collections and vector storage.
    """
    client: chromadb.HttpClient
    
    def __init__(self):
        
        self.logger = logging.getLogger(__name__)
        self.client = chromadb.HttpClient(host='localhost', port=8000,
            settings=Chroma_Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials="test-token"
            )
        )
        # Check if chroma is alive
        self.logger.debug("Heartbeat: %s", self.client.heartbeat())
        
    async def _get_user_collection_name(self, user_id: str) -> str:
        """Generate a consistent UUID-based collection name from user_id"""
        # Create a UUID from the user_id string
        user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
        return f"user_{user_uuid}"   
    
    async def get_or_create_chroma_collection(self, user_id: str):
        """Get or create chroma collection with a unique name for each user"""
        try:
            collection_name = await self._get_user_collection_name(user_id)
            chroma_collection = self.client.get_or_create_collection(name=collection_name)
            self.logger.info(f"Collection {chroma_collection} is ready")
            return chroma_collection
        except Exception as e:
            self.logger.error(f"There is an error while getting collection {e}")
    
    async def delete_collection(self, user_id: str):
        """Delete chroma collection with a unique namw(user_id)"""    
        try:
            chroma_collection = await self.get_or_create_chroma_collection(user_id=user_id)
            self.client.delete_collection(name=chroma_collection.name)
            self.logger.info(f"Delete collection: {chroma_collection}")
        except Exception as e:
            self.logger.info(f"There is an error {e}") 
    
    async def list_collection(self) -> List[str]:
        """List all collecionh that existing in chroma database"""
        try:
            collections = self.client.list_collections()
            # From Chroma Collection to list
            collection_names = [col.name for col in collections]
            self.logger.debug(f"List collections: {collection_names}")
            return collection_names
        except Exception as e:
            self.logger.error(e)
            return []
        
    async def clear_collection(self, user_id: str):
        """Clear collection so this collection will be empty"""
        try:
            chroma_collection = await self.get_or_create_chroma_collection(user_id=user_id)
            all_ids = chroma_collection.get()["ids"]  # get al docs ids
            chroma_collection.delete(ids=all_ids)
            self.logger.debug(f"Deleted all collection content: {chroma_collection}")
        except Exception as e:
            self.logger.info(f"There is an error {e}") 