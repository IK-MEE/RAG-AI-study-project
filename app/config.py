from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "ollama"
    anthropic_api_key: str = ""
    ollama_model: str = "llama3"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    chroma_db_path: str = "./chroma_db"
    data_path: str = "./data"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
