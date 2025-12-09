"""Settings for rag fast api and system"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class RagSettings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=Path("./.rag.env"), env_file_encoding="utf-8", extra="allow"
    )
    
    host: str 
    port: str
    secret_key: str
    
    db_conn_string: str

    model_api_key: str
    model_api: str
    model_name: str
    
    emb_api_key: str
    emb_model: str
    emb_api: str
    
    ranker_model: str
    ranker_api: str
    
    upload_files_dir: str
    rag_prompt_path: str = "src/settings/prompts/system_prompt.yml"
    rag_system_prompt: str = "system_common_prompt"
    
    summary_prompt_path: str = "src/settings/prompts/summary_prompt.yml"
    
    chunk_size: int = 1024
    chunk_overlap: int = 20
    
    golden_answer_path: str = "src/settings/prompts/golden_answers_kam.json"