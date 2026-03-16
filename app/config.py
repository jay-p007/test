from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str

    # Fixed app defaults (not read from .env)
    embedding_model: ClassVar[str] = "gemini-embedding-001"
    low_cost_models: ClassVar[str] = "gemini-2.5-flash-lite,gemini-2.5-flash"
    balanced_models: ClassVar[str] = "gemini-2.5-flash,gemini-2.5-flash-lite"
    high_quality_models: ClassVar[str] = "gemini-2.5-pro,gemini-2.5-flash"
    ocr_model: ClassVar[str] = "gemini-2.5-flash-lite"
    chroma_path: ClassVar[str] = "data/chroma"
    chroma_collection: ClassVar[str] = "document_chunks"
    history_max_turns: ClassVar[int] = 10
    chunk_size: ClassVar[int] = 900
    chunk_overlap: ClassVar[int] = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
