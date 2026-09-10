"""Backend configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EARLYDX_", env_file=".env", extra="ignore"
    )

    env: str = "development"
    log_level: str = "INFO"

    # If unset, the app falls back to a local SQLite file so it can run without
    # Postgres. Production/compose supplies a postgresql+psycopg URL.
    database_url: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    models_dir: str = "models"
    data_dir: str = "data"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(REPO_ROOT / 'earlydx.db').as_posix()}"

    @property
    def models_path(self) -> Path:
        p = Path(self.models_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
