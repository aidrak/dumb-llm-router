import json
import os
from typing import Any, Dict, Optional

import yaml


class Settings:
    def __init__(self):
        # API Keys (from environment)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.ollama_host = os.getenv("OLLAMA_API_HOST")
        self.openwebui_url = os.getenv("OPENWEBUI_URL")

        # Config paths
        self.config_path = os.getenv("CONFIG_PATH", "/config")
        self.model_config_file = os.getenv("MODEL_CONFIG_FILE", "models.json")
        self.routing_config_file = os.path.join(self.config_path, "config.yaml")

        # Cache for hot-reloading
        self._last_config_mtime = 0

        # Load configurations
        self.model_configs = self._load_model_configs()
        self._load_routing_config()

    def _load_model_configs(self) -> Dict[str, Any]:
        """Load model configuration from JSON file"""
        config_file_path = os.path.join(self.config_path, self.model_config_file)
        try:
            with open(config_file_path, "r") as f:
                config_data = json.load(f)
                return config_data.get("models", {})
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
                f"Warning: Config file not found at {self.routing_config_file}, using environment variables"
            )
            self._update_from_env()
        except Exception as e:
            print(f"Error loading routing config: {e}, using environment variables")
            self._update_from_env()

    def _update_from_config(self, config: Dict[str, Any]):
        """Update properties from YAML config"""
        # Model routing
        routing = config.get("routing", {})
        self.primary_model = routing.get("primary_model")
        self.fallback_model = routing.get("fallback_model")

        # If any are None, fall back to environment variables
        if not all([self.primary_model, self.fallback_model]):
            print("Some model names missing from config, checking environment variables...")
            self._fill_missing_from_env()

        # Simple thresholds
        context_detection = config.get("context_detection", {})
        self.character_length_threshold = context_detection.get("character_length_threshold", 1500)
        self.token_usage_threshold = context_detection.get("token_usage_threshold", 4000)

        # Logging
        logging_config = config.get("logging", {})
        self.log_level = logging_config.get("level", "INFO")
        self.enable_detailed_routing_logs = logging_config.get("enable_detailed_routing_logs", True)

    def _update_from_env(self):
        """Fallback to environment variables"""
        self.primary_model = os.getenv("PRIMARY_MODEL")
        self.fallback_model = os.getenv("FALLBACK_MODEL")

        # Apply final defaults only if environment variables are also missing
        self._fill_missing_from_env()

        # Other settings
        self.character_length_threshold = 1500
        self.token_usage_threshold = 4000
        self.log_level = "INFO"
        self.enable_detailed_routing_logs = True

    def _fill_missing_from_env(self):
        """Fill in any missing values with environment variables or fail if not configured"""
        self.primary_model = self.primary_model or os.getenv("PRIMARY_MODEL")
        self.fallback_model = self.fallback_model or os.getenv("FALLBACK_MODEL")

        # Check for missing required configurations
        missing = []
        if not self.primary_model:
            missing.append("primary_model")
        if not self.fallback_model:
            missing.append("fallback_model")

        if missing:
            raise ValueError(
                f"Missing required model configurations: {', '.join(missing)}. "
                f"Please configure these in config.yaml or set environment variables: "
                f"{', '.join([f'{name.upper()}' for name in missing])}"
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
