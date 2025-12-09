
import ssl
import httpx
from typing import List
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

class CustomTextEmbeddingsInference(TextEmbeddingsInference):
    """Method overriding"""
    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                auth_token = self.auth_token(self.base_url)
            else:
                auth_token = self.auth_token
            headers["Authorization"] = f"Bearer {auth_token}"
        
        json_data = {"inputs": texts, "truncate": self.truncate_text}

        async with httpx.AsyncClient(verify=ssl_context) as client:
            response = await client.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )
        return response.json()
