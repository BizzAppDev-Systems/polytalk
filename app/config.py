# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Configuration loader for PolyTalk application.

Loads configuration from YAML file and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from .ui_locales import SUPPORTED_UI_LOCALE_CODES


class Config:
    """
    Application configuration manager.

    Loads configuration from config.yaml and environment variables.
    Environment variables in config values are expanded using ${VAR_NAME} syntax.
    """

    _instance: "Config | None" = None

    def __new__(cls) -> "Config":
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration if not already initialized."""
        if self._initialized:
            return

        # Load .env file
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        # Load config file
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            config_path = (
                Path(__file__).parent.parent / "config" / "config.yaml.example"
            )

        if config_path.exists():
            with open(config_path, "r") as f:
                self._raw_config = yaml.safe_load(f)
        else:
            self._raw_config = {}

        # Expand environment variables in config
        self._config = self._expand_env_vars(self._raw_config)

        self._initialized = True

    def _expand_env_vars(self, obj: Any) -> Any:
        """
        Recursively expand environment variables in configuration.

        Args:
            obj: Configuration object (dict, list, or string)

        Returns:
            Configuration with environment variables expanded
        """
        if isinstance(obj, str):
            return self._expand_string(obj)
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        else:
            return obj

    def _expand_string(self, value: str) -> Any:
        """
        Expand environment variables in a string value.

        Args:
            value: String value that may contain ${VAR_NAME} patterns

        Returns:
            String with environment variables replaced, or original value if not a template
        """
        import re

        pattern = r"\$\{([^}]+)\}"

        def replace(match: Any) -> str:
            var_name = match.group(1)
            # Get value from environment, return original if not found
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            return match.group(0)

        result = re.sub(pattern, replace, value)

        # If no expansion happened, try to convert to appropriate type
        if result == value:
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            elif value.isdigit():
                return int(value)
            elif value.replace(".", "").isdigit() and value.count(".") == 1:
                return float(value)

        return result

    @property
    def whisper(self) -> dict:
        """Get Whisper configuration."""
        return self._config.get("whisper", {})

    @property
    def vad(self) -> dict:
        """Get shared voice activity detection configuration."""
        return self._config.get("vad", {})

    @property
    def translation(self) -> dict:
        """Get translation AI configuration."""
        return self._config.get("translation", {})

    @property
    def visual_context(self) -> dict:
        """Get shared tab/page visual context configuration."""
        return self._config.get("visual_context", {})

    @property
    def tts(self) -> dict:
        """Get TTS configuration."""
        return self._config.get("tts", {})

    @property
    def app(self) -> dict:
        """Get application configuration."""
        return self._config.get("app", {})

    @property
    def media_output_dir(self) -> Path:
        """Get media output directory path."""
        return Path(self.app.get("media_output_dir", "media/output"))

    @property
    def debug(self) -> bool:
        """Get debug mode setting."""
        return bool(self.app.get("debug", False))

    @property
    def host(self) -> str:
        """Get application host."""
        return str(self.app.get("host", "0.0.0.0"))

    @property
    def port(self) -> int:
        """Get application port."""
        value = self.app.get("port", 8000)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 8000

    @property
    def default_ui_locale(self) -> str:
        """Get the default UI locale."""
        value = os.environ.get("DEFAULT_UI_LOCALE") or self.app.get(
            "default_ui_locale", "en"
        )
        normalized = str(value).strip().replace("-", "_").split("_", 1)[0]
        return normalized if normalized in SUPPORTED_UI_LOCALE_CODES else "en"

    @property
    def supported_ui_locales(self) -> list[str]:
        """Get supported UI locales for the web interface."""
        configured = os.environ.get("SUPPORTED_UI_LOCALES") or self.app.get(
            "supported_ui_locales"
        )
        if configured:
            if isinstance(configured, str):
                values = [value.strip() for value in configured.split(",")]
            else:
                values = [str(value).strip() for value in configured]
            locales = [
                value.replace("-", "_").split("_", 1)[0] for value in values if value
            ]
            filtered = [
                locale for locale in locales if locale in SUPPORTED_UI_LOCALE_CODES
            ]
            if filtered:
                return filtered
        return sorted(SUPPORTED_UI_LOCALE_CODES)


def get_config() -> Config:
    """
    Get the singleton configuration instance.

    Returns:
        Config instance
    """
    return Config()
