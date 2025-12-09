import httpx
from llama_index.core import Settings
from llama_index.llms.openllm import OpenLLM

import logging
from src.services.embedding_inference import CustomTextEmbeddingsInference
from config import RagSettings

logger = logging.getLogger(__name__)

settings = RagSettings()

_initialized = False

async def messages_to_prompt(messages):
    """Correct message formatting"""
    formatted = []
    for message in messages:
        if message.role == "system":
            formatted.append(f"<|im_start|>system\n{message.content}<|im_end|>")
        elif message.role == "user":
            formatted.append(f"<|im_start|>user\n{message.content}<|im_end|>")
        elif message.role == "assistant":
            formatted.append(f"<|im_start|>assistant\n{message.content}<|im_end|>")
    
    return "\n".join(formatted)

async def completion_to_prompt(completion):
    return f"<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

async def initialize_settings(aclient: httpx.AsyncClient, client: httpx.Client):
    
    global _initialized
    if not _initialized:
        try:
            logger.debug("Start to load models...")
            Settings.embed_model = CustomTextEmbeddingsInference(
                    model_name=settings.emb_model,  # MUST BE THE SAME AS IN OFFICIAL DOCS
                    base_url=settings.emb_api,
                    auth_token=settings.emb_api_key,
                    timeout=60,  
                    embed_batch_size=10,  # batch size for embedding
                )
            logger.debug("Emdeddig model is ready")
            Settings.llm = OpenLLM(
                model=settings.model_name,
                api_base=settings.model_api,
                api_key=settings.model_api_key,
                temperature=0.0,
                is_chat_model=True,
                is_function_calling_model=True,
                context_window=32768,
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                async_http_client=aclient,
                http_client=client,
                timeout=140
            )
            logger.debug("Qwen model is ready to response")

        except Exception as e:
            logger.error(f"There is an error while loading models: {e}")
            raise
        _initialized = True