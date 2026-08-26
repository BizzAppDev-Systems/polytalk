"""Tests for configuration loading."""

from pathlib import Path
from unittest.mock import patch


from app.config import Config, get_config


class TestConfig:
    """Tests for configuration loading."""

    def test_config_singleton(self):
        """Test that config is a singleton."""
        config1 = Config()
        config2 = Config()

        assert config1 is config2

    def test_config_has_whisper_section(self):
        """Test that config has whisper section."""
        config = Config()

        assert "whisper" in config._config
        assert config._config["whisper"].get("base_url") is not None

    def test_config_has_translation_section(self):
        """Test that config has translation section."""
        config = Config()

        assert "translation" in config._config
        assert config._config["translation"].get("base_url") is not None

    def test_config_has_visual_context_section(self):
        """Test that config exposes visual context settings."""
        config = Config()

        assert config.visual_context is not None

    def test_config_has_tts_section(self):
        """Test that config has TTS section."""
        config = Config()

        assert config.tts is not None
        assert "provider" in config.tts

    def test_config_has_app_section(self):
        """Test that config has app section."""
        config = Config()

        assert config.app is not None
        assert "host" in config.app

    def test_parse_bool_config_values(self):
        """Test shared boolean config parsing helper."""
        from app.utils.config import parse_bool_config

        assert parse_bool_config(True, False) is True
        assert parse_bool_config(1, False) is True
        assert parse_bool_config(0, True) is False
        assert parse_bool_config("yes", False) is True
        assert parse_bool_config("OFF", True) is False
        assert parse_bool_config(2, True) is True
        assert parse_bool_config("unexpected", True) is True

    def test_parse_float_config_values(self):
        """Test shared floating-point config parsing helper."""
        from app.utils.config import parse_float_config

        assert parse_float_config(12.5, 30.0) == 12.5
        assert parse_float_config("12.5", 30.0) == 12.5
        assert parse_float_config("${STT_FALLBACK_REPLAY_MAX_SECONDS}", 30.0) == 30.0
        assert (
            parse_float_config("${STT_FALLBACK_REPLAY_MAX_SECONDS:-30.0}", 30.0) == 30.0
        )
        assert parse_float_config(None, 30.0) == 30.0
        assert parse_float_config(True, 30.0) == 30.0

    def test_parse_int_config_values(self):
        """Test shared integer config parsing helper."""
        from app.utils.config import parse_int_config

        assert parse_int_config(123, 250) == 123
        assert parse_int_config("123", 250) == 123
        assert parse_int_config("${CUSTOM_INSTRUCTION_MAX_CHARS}", 250) == 250
        assert parse_int_config(None, 250) == 250
        assert parse_int_config(True, 250) == 250

    def test_custom_instruction_max_chars_cache_warns_once(self, caplog):
        """Custom instruction limit parsing is cached per raw config value."""
        import logging

        from app.utils.config import (
            clear_custom_instruction_max_chars_cache,
            get_custom_instruction_max_chars,
        )

        clear_custom_instruction_max_chars_cache()
        with caplog.at_level(logging.WARNING, logger="app.utils.config"):
            assert get_custom_instruction_max_chars("250MB") == 250
            assert get_custom_instruction_max_chars("250MB") == 250

        assert caplog.text.count("Invalid integer config") == 1
        clear_custom_instruction_max_chars_cache()

    def test_parse_int_config_warns_for_invalid_values(self, caplog):
        """Invalid concrete integer config values emit a warning."""
        import logging

        from app.utils.config import parse_int_config

        with caplog.at_level(logging.WARNING, logger="app.utils.config"):
            assert (
                parse_int_config(
                    "250MB",
                    250,
                    name="custom_instruction_max_chars",
                    warn_on_invalid=True,
                )
                == 250
            )

        assert "custom_instruction_max_chars" in caplog.text
        assert "250MB" in caplog.text

    def test_parse_int_config_does_not_warn_for_missing_values(self, caplog):
        """Missing integer config values fall back without noisy warnings."""
        import logging

        from app.utils.config import parse_int_config

        with caplog.at_level(logging.WARNING, logger="app.utils.config"):
            assert (
                parse_int_config(
                    None,
                    250,
                    name="custom_instruction_max_chars",
                    warn_on_invalid=True,
                )
                == 250
            )

        assert caplog.text == ""

    def test_parse_int_config_does_not_warn_for_unresolved_placeholders(self, caplog):
        """Unresolved ${VAR} placeholders fall back without noisy warnings."""
        import logging

        from app.utils.config import parse_int_config

        with caplog.at_level(logging.WARNING, logger="app.utils.config"):
            assert (
                parse_int_config(
                    "${CUSTOM_INSTRUCTION_MAX_CHARS}",
                    250,
                    name="custom_instruction_max_chars",
                    warn_on_invalid=True,
                )
                == 250
            )

        assert caplog.text == ""

    def test_config_env_expansion(self):
        """Test environment variable expansion."""
        import os

        os.environ["TTS_PROVIDER"] = "piper"
        os.environ["TRANSLATION_BASE_URL"] = "https://test.ai.com"
        os.environ["WHISPER_BASE_URL"] = "https://test.whisper.com"

        Config._instance = None
        config = Config()

        tts_provider = config.tts.get("provider", "")
        trans_base_url = config.translation.get("base_url", "")
        whisper_base_url = config.whisper.get("base_url", "")

        assert not str(tts_provider).startswith("${")
        assert not str(trans_base_url).startswith("${")
        assert not str(whisper_base_url).startswith("${")

    def test_config_type_conversion(self):
        """Test that config values are properly typed."""
        import os

        os.environ["APP_PORT"] = "8000"
        os.environ["APP_DEBUG"] = "true"
        os.environ["APP_HOST"] = "0.0.0.0"

        Config._instance = None
        config = Config()

        assert config.port is not None
        assert isinstance(config.port, int)
        assert config.debug is not None
        assert isinstance(config.debug, bool)
        assert isinstance(config.host, str)

    def test_config_port_falls_back_for_unresolved_env_placeholder(self):
        """Test unresolved APP_PORT placeholders do not break app startup."""
        config = Config()
        config._config = {"app": {"port": "${APP_PORT}"}}

        assert config.port == 8000

    def test_get_config_function(self):
        """Test get_config function returns Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_config_properties(self):
        """Test config property accessors."""
        config = Config()

        assert config.whisper is not None
        assert config.translation is not None
        assert config.tts is not None
        assert config.app is not None

    def test_config_media_output_dir_path(self):
        """Test media output directory returns Path object."""
        config = Config()
        assert isinstance(config.media_output_dir, Path)

    def test_config_with_file_not_found(self):
        """Test config handles file not found."""
        with patch("app.config.os.path.exists", return_value=False):
            Config._instance = None
            config = Config()
            assert config is not None
