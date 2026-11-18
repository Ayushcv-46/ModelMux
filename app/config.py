import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        gemini_api_key: str = ""
        ollama_base_url: str = "http://localhost:11434"

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    settings = Settings()

except ImportError:
    # Graceful fallback until `pip install pydantic-settings` is run
    class Settings:  # type: ignore
        def __init__(self):
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())

            self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
            self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    settings = Settings()
