"""
Configuration management for CiteWright.

Handles loading and saving configuration from files and environment variables.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


APP_NAME = "citewright"
CONFIG_DIR = Path.home() / ".config" / APP_NAME
CACHE_DIR = Path.home() / ".cache" / APP_NAME
LOG_DIR = Path.home() / ".local" / "share" / APP_NAME
CONFIG_FILE = CONFIG_DIR / "config.json"

# Ensure directories exist
for directory in [CONFIG_DIR, CACHE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Supported LLM providers with their configurations
AI_PROVIDERS = {
    "ollama": {
        "env_key": None,
        "base_url": "http://localhost:11434/v1",
        "text_model": "llama3.2",
        "vision_model": "llava",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "text_model": "gpt-4o-mini",
        "vision_model": "gpt-4o",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
        "text_model": "claude-3-5-haiku-20241022",
        "vision_model": "claude-3-5-sonnet-20241022",
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "text_model": "gemini-2.0-flash",
        "vision_model": "gemini-2.0-flash",
    },
}


@dataclass
class Config:
    """Application configuration."""
    
    # LLM settings
    ai_provider: str = "ollama"
    ai_enabled: bool = False
    
    # Rename settings
    max_title_words: int = 5
    filename_format: str = "{author}_{year}_{title}"
    dry_run: bool = True
    recursive: bool = False
    
    # API settings
    unpaywall_email: str = "anonymous@example.com"
    semantic_scholar_api_key: Optional[str] = None
    
    # Output settings
    generate_bibtex: bool = True
    bibtex_file: str = "references.bib"
    
    # Logging
    log_level: str = "INFO"
    
    # Supported extensions
    pdf_extensions: list = field(default_factory=lambda: [".pdf"])
    media_extensions: list = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".heic", ".mp4"
    ])

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from file and environment variables."""
        config_data = {}
        
        # Load from file if exists
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    config_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Override with environment variables
        env_mappings = {
            "CITEWRIGHT_AI_PROVIDER": "ai_provider",
            "CITEWRIGHT_AI_ENABLED": "ai_enabled",
            "CITEWRIGHT_DRY_RUN": "dry_run",
            "UNPAYWALL_EMAIL": "unpaywall_email",
            "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle boolean conversion
                if config_key in ("ai_enabled", "dry_run", "recursive", "generate_bibtex"):
                    config_data[config_key] = value.lower() in ("true", "1", "yes")
                else:
                    config_data[config_key] = value
        
        return cls(**{k: v for k, v in config_data.items() if hasattr(cls, k)})
    
    def save(self) -> None:
        """Save configuration to file."""
        config_data = {
            "ai_provider": self.ai_provider,
            "ai_enabled": self.ai_enabled,
            "max_title_words": self.max_title_words,
            "filename_format": self.filename_format,
            "dry_run": self.dry_run,
            "recursive": self.recursive,
            "unpaywall_email": self.unpaywall_email,
            "generate_bibtex": self.generate_bibtex,
            "bibtex_file": self.bibtex_file,
            "log_level": self.log_level,
        }
        
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=2)
    
    def get_ai_config(self, vision: bool = False) -> Dict[str, Any]:
        """Get LLM provider configuration.
        
        Args:
            vision: If True, return vision-capable model config.
            
        Returns:
            Dictionary with provider, api_key, base_url, and model.
        """
        provider = self.ai_provider.lower()
        if provider not in AI_PROVIDERS:
            provider = "ollama"
        
        provider_config = AI_PROVIDERS[provider]
        
        # Get API key from environment
        api_key = None
        if provider_config["env_key"]:
            api_key = os.getenv(provider_config["env_key"])
        elif provider == "ollama":
            api_key = "ollama"  # Placeholder for Ollama
        
        model_key = "vision_model" if vision else "text_model"
        
        return {
            "provider": provider,
            "api_key": api_key,
            "base_url": provider_config["base_url"],
            "model": provider_config[model_key],
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config() -> Config:
    """Reload configuration from disk."""
    global _config
    _config = Config.load()
    return _config
