import os
from pathlib import Path
from pydantic import BaseModel, Field
import yaml

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        gemini_api_key: str = ""
        groq_api_key: str = ""
        ollama_base_url: str = "http://localhost:11434"
        routing_priority: str = "groq,gemini,ollama"

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        @property
        def priority_list(self) -> list[str]:
            """Parse comma-separated routing_priority into a clean list of provider names."""
            return [p.strip().lower() for p in self.routing_priority.split(",") if p.strip()]

    settings = Settings()

except ImportError:
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
            self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
            self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.routing_priority: str = os.getenv("ROUTING_PRIORITY", "groq,gemini,ollama")

        @property
        def priority_list(self) -> list[str]:
            return [p.strip().lower() for p in self.routing_priority.split(",") if p.strip()]

    settings = Settings()


# ============================================================================
# YAML Configuration Models (Day 19: Config-driven design)
# ============================================================================

class RoutingConfig(BaseModel):
    strategy: str = "priority"
    priority: list[str] = Field(default_factory=lambda: ["groq", "gemini", "ollama"])


class ReliabilityConfig(BaseModel):
    total_timeout_seconds: float = 30.0
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0


class ProviderConfig(BaseModel):
    timeout_seconds: float = 30.0
    models: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    def get_provider_timeout(self, provider_name: str, default: float = 30.0) -> float:
        """Return configured timeout_seconds for a given provider, or default if not set."""
        prov_cfg = self.providers.get(provider_name.lower())
        if prov_cfg and prov_cfg.timeout_seconds > 0:
            return prov_cfg.timeout_seconds
        return default

    @property
    def model_to_provider(self) -> dict[str, str]:
        """
        Builds a reverse lookup mapping from lowercase model name -> lowercase provider name.
        e.g., {"llama-3.3-70b-versatile": "groq", "gemini-1.5-flash": "gemini"}
        """
        mapping: dict[str, str] = {}
        for provider_name, prov_cfg in self.providers.items():
            for model in prov_cfg.models:
                mapping[model.lower()] = provider_name.lower()
        return mapping


def load_yaml_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Read and validate config.yaml. Falls back to default AppConfig if file is missing."""
    p = Path(config_path)
    if not p.is_file():
        return AppConfig()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    except Exception:
        return AppConfig()


# Global config instance loaded at startup
yaml_config = load_yaml_config()
