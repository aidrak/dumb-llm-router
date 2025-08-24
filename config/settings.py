import json
import os
from typing import Any, Dict, Optional

import yaml


class Settings:
    # Default configuration values
    DEFAULTS = {
        "working_model": "working_model",
        "searching_model": "searching_model", 
        "fallback_model": "working_model",
        "search_keywords": [],
        "character_length_threshold": 1500,
        "token_usage_threshold": 4000,
        "log_level": "INFO",
        "enable_detailed_routing_logs": True,
    }

    def __init__(self):
        # API Keys (from environment) - Ensure virtual environment is used
        if not os.getenv("VIRTUAL_ENV"):
            print("⚠️  Warning: Virtual environment not detected. Please use a virtual environment for security.")
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.ollama_host = os.getenv("OLLAMA_API_HOST")
        self.openwebui_url = os.getenv("OPENWEBUI_URL")
        
        # RAG Configuration
        self.use_gemini_rag = os.getenv("USE_GEMINI_RAG", "false").lower() == "true"

        # Config paths
        self.config_path = os.getenv("CONFIG_PATH", "/config")
        self.model_config_file = os.getenv("MODEL_CONFIG_FILE", "models.json")
        self.routing_config_file = os.path.join(self.config_path, "config.yaml")

        # Cache for hot-reloading
        self._last_config_mtime = 0

        # Apply defaults first
        self._apply_defaults()

        # Load configurations
        self.model_configs = self._load_model_configs()
        self._load_routing_config()
        self._validate_model_configs()

    def _apply_defaults(self):
        """Apply default configuration values"""
        for key, value in self.DEFAULTS.items():
            setattr(self, key, value)
        
        # Backward compatibility
        self.primary_model = self.working_model

    def _load_model_configs(self) -> Dict[str, Any]:
        """Load model configuration from JSON file - new format only"""
        config_file_path = os.path.join(self.config_path, self.model_config_file)
        try:
            with open(config_file_path, "r") as f:
                config_data = json.load(f)
                # Only support new format - root cause fix for complexity
                if "models" in config_data:
                    raise ValueError(
                        "Old 'models' format detected in models.json. "
                        "Please migrate to new format with 'working_model' and 'searching_model' at root level."
                    )
                return config_data
        except Exception as e:
            print(f"Error loading model config: {e}")
            return {}

    def _load_routing_config(self):
        """Load routing config with hot-reload capability"""
        try:
            current_mtime = os.path.getmtime(self.routing_config_file)

            # Only reload if file changed or first load
            if current_mtime != self._last_config_mtime:
                print(f"🔄 Loading routing configuration from {self.routing_config_file}")

                with open(self.routing_config_file, "r") as f:
                    config = yaml.safe_load(f) or {}

                self._last_config_mtime = current_mtime
                self._update_from_config(config)
                print("✅ Routing configuration loaded")

        except FileNotFoundError:
            print(
                f"Warning: Config file not found at {self.routing_config_file}, "
                "using default values"
            )
        except Exception as e:
            print(f"Error loading routing config: {e}, using default values")

    def _update_from_config(self, config: Dict[str, Any]):
        """Update properties from YAML config - single source of truth"""
        # Override defaults with config values where available
        routing = config.get("routing", {})
        self.search_keywords = routing.get("search_keywords", self.DEFAULTS["search_keywords"])

        # Simple thresholds
        context_detection = config.get("context_detection", {})
        self.character_length_threshold = context_detection.get(
            "character_length_threshold", self.DEFAULTS["character_length_threshold"]
        )
        self.token_usage_threshold = context_detection.get(
            "token_usage_threshold", self.DEFAULTS["token_usage_threshold"] 
        )

        # Logging
        logging_config = config.get("logging", {})
        self.log_level = logging_config.get("level", self.DEFAULTS["log_level"])
        self.enable_detailed_routing_logs = logging_config.get(
            "enable_detailed_routing_logs", self.DEFAULTS["enable_detailed_routing_logs"]
        )

    def _validate_model_configs(self):
        """Validate that required models exist in model_configs"""
        missing = []
        if "working_model" not in self.model_configs:
            missing.append("working_model")
        if "searching_model" not in self.model_configs:
            missing.append("searching_model")

        if missing:
            raise ValueError(
                f"Missing required model definitions in models.json: {', '.join(missing)}. "
                f"Please define these models in your models.json file."
            )

    def reload_if_changed(self):
        """Check if config changed and reload if needed"""
        self._load_routing_config()

    def read_system_prompt_from_file(self, file_path_relative_to_config: str) -> Optional[str]:
        """Read system prompt content from file"""
        full_file_path = os.path.join(self.config_path, file_path_relative_to_config)
        try:
            with open(full_file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read prompt file {full_file_path}: {e}")
            return None


# Global settings instance
settings = Settings()
