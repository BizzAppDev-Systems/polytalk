# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for translation service methods.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.translation_service import TranslationService


class TestSemanticBoundarySelection:
    """Test the Qwen-only sentence-boundary selector contract."""

    @pytest.mark.asyncio
    async def test_selects_original_text_prefix_for_full_sentence_and_fragment(self):
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "http://fallback:8001",
                "endpoint": "/v1/chat/completions",
                "api_format": "openai_chat",
                "model": "fallback-model",
                "semantic_boundary_enabled": True,
                "semantic_boundary_provider": "qwen3",
                "providers": {
                    "qwen3": {
                        "base_url": "http://qwen:8001",
                        "endpoint": "/v1/chat/completions",
                        "api_format": "openai_chat",
                        "model": "polytalk-vllm",
                    }
                },
            }
            service = TranslationService()

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "boundary_id": 4,
                                "confidence": 0.96,
                                "reason": "new clause follows",
                            }
                        )
                    }
                }
            ]
        }
        with (
            patch.object(
                service._http_client, "post", new=AsyncMock(return_value=response)
            ) as post,
            patch("app.services.translation_service.logger.debug") as debug_log,
        ):
            decision = await service.select_semantic_boundary(
                "Meeting is on Friday Do not forget to", "en"
            )

        assert decision is not None
        assert decision.boundary == len("Meeting is on Friday")
        assert post.await_args.args[0] == "http://qwen:8001/v1/chat/completions"
        payload = post.await_args.kwargs["json"]
        assert payload["model"] == "polytalk-vllm"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "Friday [4] Do [5]" in payload["messages"][1]["content"]
        debug_log.assert_called_once_with(
            "Semantic boundary chat template: provider=%s enable_thinking=%s",
            "qwen3",
            False,
        )
        await service.close()

    @pytest.mark.asyncio
    async def test_uses_unicode_character_offset_for_gujarati_prefix(self):
        """Gujarati boundaries use Python character offsets, not UTF-8 byte offsets."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "api_format": "openai_chat",
                "semantic_boundary_enabled": True,
            }
            service = TranslationService()

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"boundary_id": 3, "confidence": 0.96})
                    }
                }
            ]
        }
        with patch.object(
            service._http_client, "post", new=AsyncMock(return_value=response)
        ):
            decision = await service.select_semantic_boundary(
                "\u0aac\u0ac7\u0aa0\u0a95 \u0ab6\u0ac1\u0a95\u0acd\u0ab0\u0ab5\u0abe\u0ab0\u0ac7 \u0a9b\u0ac7 \u0aad\u0ac2\u0ab2\u0ab6\u0acb \u0aa8\u0ab9\u0ac0\u0a82",
                "gu",
            )

        expected_prefix = "\u0aac\u0ac7\u0aa0\u0a95 \u0ab6\u0ac1\u0a95\u0acd\u0ab0\u0ab5\u0abe\u0ab0\u0ac7 \u0a9b\u0ac7"
        assert decision is not None
        assert decision.boundary == len(expected_prefix)
        assert len(expected_prefix.encode("utf-8")) > decision.boundary
        await service.close()

    @pytest.mark.asyncio
    async def test_semantic_boundary_is_enabled_when_config_is_omitted(self):
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
            }
            service = TranslationService()

        assert service.semantic_boundary_enabled is True
        await service.close()

    @pytest.mark.asyncio
    async def test_no_boundary_returns_none_and_can_omit_chat_template_kwargs(self):
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "api_format": "openai_chat",
                "semantic_boundary_enabled": True,
                "semantic_boundary_chat_template_kwargs_enabled": False,
            }
            service = TranslationService()

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"boundary_id": 0, "confidence": 0.99})
                    }
                }
            ]
        }
        with patch.object(
            service._http_client, "post", new=AsyncMock(return_value=response)
        ) as post:
            decision = await service.select_semantic_boundary("Do not forget to", "en")

        assert decision is None
        assert "chat_template_kwargs" not in post.await_args.kwargs["json"]
        await service.close()

    @pytest.mark.asyncio
    async def test_rejects_low_confidence_boundary(self):
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "api_format": "openai_chat",
                "semantic_boundary_enabled": True,
                "semantic_boundary_min_confidence": 0.75,
            }
            service = TranslationService()

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"boundary_id": 4, "confidence": 0.40})
                    }
                }
            ]
        }
        with patch.object(
            service._http_client, "post", new=AsyncMock(return_value=response)
        ):
            decision = await service.select_semantic_boundary(
                "Meeting is on Friday Do not forget to", "en"
            )

        assert decision is None
        await service.close()


class TestTranslationServiceClose:
    """Test translation service close."""

    @pytest.mark.asyncio
    async def test_close_service(self):
        """Test closing translation service."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": True,
            }
            service = TranslationService()

            with patch.object(service._http_client, "aclose", new_callable=AsyncMock):
                await service.close()


class TestTranslationServiceRealTranslateErrorHandling:
    """Test real translation error handling paths."""

    @pytest.mark.asyncio
    async def test_real_translate_json_parse_error(self):
        """Test translation with JSON parse error."""
        from unittest.mock import MagicMock

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "invalid json"
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status.return_value = None

            with patch.object(service._http_client, "post", return_value=mock_response):
                await service.translate("Hello", "en", "gu")

                call_args = service._http_client.post.call_args
                assert call_args[1]["json"]["max_tokens"] == 240

    @pytest.mark.asyncio
    async def test_mock_translate_different_language_pairs(self):
        """Test mock translation with different language pairs."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": True,
            }
            service = TranslationService()

            test_cases = [
                ("en", "gu", "તમે કેમ છો?"),
                ("en", "hi", "आप कैसे हैं?"),
                ("en", "es", "¿Cómo estás?"),
                ("en", "fr", "Comment allez-vous?"),
                ("en", "de", "Wie geht es Ihnen?"),
                ("gu", "en", "How are you?"),
                ("hi", "en", "How are you?"),
                ("es", "en", "How are you?"),
                ("fr", "en", "How are you?"),
                ("de", "en", "How are you?"),
            ]

            for source, target, expected in test_cases:
                result = service._mock_translate("How are you?", source, target)
                assert result.success is True
                assert result.text == expected

    @pytest.mark.asyncio
    async def test_mock_translate_unknown_language_pair(self):
        """Test mock translation with unknown language pair."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": True,
            }
            service = TranslationService()

            result = service._mock_translate("Hello", "xx", "yy")
            assert result.success is True
            assert result.text == "[yy] Hello"

    @pytest.mark.asyncio
    async def test_translate_disabled_service(self):
        """Test translation when service is disabled."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": False,
                "mock_mode": True,
            }
            service = TranslationService()

            result = await service.translate("Hello", "en", "gu")

            assert result.success is False
            assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_translate_real_mode_timeout(self):
        """Test real translation with timeout."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            with patch.object(
                service._http_client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_post.side_effect = httpx.TimeoutException("Timeout")

                result = await service.translate("Hello", "en", "gu")

                assert result.success is False
                assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_translate_real_mode_http_error(self):
        """Test real translation with HTTP error."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            with patch.object(
                service._http_client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_post.side_effect = httpx.HTTPError("HTTP Error")

                result = await service.translate("Hello", "en", "gu")

                assert result.success is False

    @pytest.mark.asyncio
    async def test_translate_retry_logic(self):
        """Test translation retry logic on failure."""
        from unittest.mock import MagicMock

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            call_count = 0

            def mock_post(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise httpx.HTTPError("Temporary error")
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "success"}}]
                }
                return mock_response

            with patch.object(service._http_client, "post", side_effect=mock_post):
                await service.translate("Hello", "en", "gu")

                assert call_count >= 1

    @pytest.mark.asyncio
    async def test_real_translate_with_invalid_max_tokens(self, caplog):
        """Test real translation handles invalid max_tokens."""
        from unittest.mock import MagicMock

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
                "max_tokens": "invalid",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            assert service.max_tokens == 240

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "result"}}]
            }
            mock_response.raise_for_status.return_value = None

            with patch.object(service._http_client, "post", return_value=mock_response):
                await service.translate("Hello", "en", "gu")

    @pytest.mark.asyncio
    async def test_real_translate_with_system_prompt_template(self):
        """Test real translation uses system prompt template."""
        from unittest.mock import MagicMock

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
                "system_prompt": "Translate from {source_language} to {target_language}",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "result"}}]
            }
            mock_response.raise_for_status.return_value = None

            with patch.object(service._http_client, "post", return_value=mock_response):
                await service.translate("Hello", "en", "gu")

                call_args = service._http_client.post.call_args
                messages = call_args[1]["json"]["messages"]
                assert "English" in messages[0]["content"]
                assert "Gujarati" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_real_translate_with_gemini_provider(self):
        """Test real translate with Gemini provider."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "gemini",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "test_key",
                "api_format": "gemini_generate_content",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "translated"}]}}]
            }

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is True

    @pytest.mark.asyncio
    async def test_real_translate_with_anthropic_provider(self):
        """Test real translate with Anthropic provider."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com",
                "api_key": "test_key",
                "api_format": "anthropic_messages",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"content": [{"text": "translated"}]}

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is True

    @pytest.mark.asyncio
    async def test_real_translate_with_openai_responses_provider(self):
        """Test real translate with OpenAI Responses API provider."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
                "use_responses_api": True,
                "api_format": "openai_responses",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": [{"content": [{"type": "output_text", "text": "translated"}]}]
            }
            mock_response.raise_for_status.return_value = None

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is True

    @pytest.mark.asyncio
    async def test_translate_with_empty_text(self):
        """Test translate with empty text."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": True,
            }

            service = TranslationService()
            result = await service.translate("", "en", "gu")

            assert result.success is True
            assert result.text == "તમે કેમ છો?"

    @pytest.mark.asyncio
    async def test_translate_with_whitespace_only(self):
        """Test translate with whitespace only."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": True,
            }

            service = TranslationService()
            result = await service.translate("   ", "en", "gu")

            assert result.success is True
            assert result.text == "તમે કેમ છો?"

    @pytest.mark.asyncio
    async def test_real_translate_with_http_status_401(self):
        """Test real translate handles HTTP 401."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "invalid_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_response.raise_for_status.side_effect = Exception(
                "401 Client Error: Unauthorized"
            )

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_real_translate_with_http_status_429(self):
        """Test real translate handles HTTP 429 rate limit."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"
            mock_response.raise_for_status.side_effect = Exception(
                "429 Client Error: Too Many Requests"
            )

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_real_translate_with_http_status_503(self):
        """Test real translate handles HTTP 503 service unavailable."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.text = "Service unavailable"
            mock_response.raise_for_status.side_effect = Exception(
                "503 Server Error: Service Unavailable"
            )

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_real_translate_with_http_status_504(self):
        """Test real translate handles HTTP 504 gateway timeout."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 504
            mock_response.text = "Gateway timeout"
            mock_response.raise_for_status.side_effect = Exception(
                "504 Server Error: Gateway Timeout"
            )

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_translate_with_malformed_json_response(self):
        """Test translate with malformed JSON response."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("test", "doc", 0)
            mock_response.raise_for_status.return_value = None

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_translate_with_missing_text_field(self):
        """Test translate with missing text field in response."""

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = {
                "enabled": True,
                "mock_mode": False,
                "provider": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "test_key",
            }
            mock_config.return_value.app = {}

            service = TranslationService()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"wrong_field": "translated"}

            with patch.object(service._http_client, "post", return_value=mock_response):
                result = await service.translate("Hello", "en", "gu")
                assert result.success is False


class TestTranslationProviderRouting:
    """Test priority-based translation provider routing."""

    def _service_config(self):
        return {
            "enabled": True,
            "mock_mode": False,
            "base_url": "https://default.example.com",
            "endpoint": "/v1/chat/completions",
            "api_format": "openai_chat",
            "api_key": "default-key",
            "model": "legacy-default",
            "default_provider": "qwen3",
            "providers": {
                "qwen3": {
                    "base_url": "https://qwen.example.com",
                    "endpoint": "/v1/chat/completions",
                    "api_format": "openai_chat",
                    "api_key": "qwen-key",
                    "model": "qwen3",
                },
                "translategamma": {
                    "base_url": "https://gamma.example.com",
                    "endpoint": "/v1/chat/completions",
                    "api_format": "translategemma_chat",
                    "api_key": "gamma-key",
                    "model": "translategamma",
                },
            },
            "routing": [
                {
                    "priority": 10,
                    "target_langs": ["gu", "hi", "mr"],
                    "provider": "translategamma",
                },
                {
                    "priority": 20,
                    "source_langs": ["en"],
                    "provider": "qwen3",
                },
            ],
        }

    def test_routing_disabled_uses_configured_default_provider(self):
        """Legacy mode ignores language routing rules and uses the default provider."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        provider = service._resolve_provider_config("en", "gu", routing_enabled=False)

        assert provider["model"] == "qwen3"

    def test_routing_enabled_selects_matching_provider(self):
        """Custom mode applies the configured language routing rules."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        provider = service._resolve_provider_config("en", "gu", routing_enabled=True)

        assert provider["model"] == "translategamma"

    def test_routing_priority_selects_highest_priority_matching_provider(self):
        """Test priority resolves source and target language rule conflicts."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        url, headers, payload = service._build_translation_request("Hello", "en", "gu")

        assert url == "https://gamma.example.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer gamma-key"
        assert payload["model"] == "translategamma"

    def test_translategemma_uses_structured_user_only_message(self):
        """TranslateGemma receives its strict language-code chat payload."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        _, _, payload = service._build_translation_request(" Hello ", "en_US", "gu_IN")

        assert payload["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en-US",
                        "target_lang_code": "gu-IN",
                        "text": "Hello",
                    }
                ],
            }
        ]

    def test_translategemma_custom_instruction_uses_qwen(self):
        """Unsupported instruction prompting stays on the default provider."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        url, _, payload = service._build_translation_request(
            "Hello", "en", "gu", custom_instruction="Use a formal tone"
        )

        assert url == "https://qwen.example.com/v1/chat/completions"
        assert payload["model"] == "qwen3"
        assert payload["messages"][0]["role"] == "system"

    def test_translategemma_missing_confidence_remains_unavailable(self):
        """TranslateGemma does not fabricate a 50% quality score."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        text, quality = service._parse_translation_response(
            {"choices": [{"message": {"content": "નમસ્તે"}}]},
            self._service_config()["providers"]["translategamma"],
        )

        assert text == "નમસ્તે"
        assert quality is None

    def test_translategemma_explanatory_output_requires_fallback(self):
        """Multiple alternatives are rejected as a non-direct translation."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        assert service._translategemma_output_requires_fallback(
            "Come on.",
            "চলুন।\nঅথবা,\nআয়।\n(প্রসঙ্গ অনুযায়ী উপযুক্ত অনুবাদ ব্যবহার করা উচিত।)",
        )
        assert not service._translategemma_output_requires_fallback(
            "Come on.", "চলুন।\nএগিয়ে যান।"
        )
        assert not service._translategemma_output_requires_fallback("Come on.", "চলুন।")

    @pytest.mark.asyncio
    async def test_translategemma_explanatory_output_retries_with_qwen(self):
        """A verbose TranslateGemma response is replaced by the local default."""
        config = self._service_config()
        config["routing"][0]["target_langs"].append("bn")
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        gamma_response = MagicMock()
        gamma_response.raise_for_status.return_value = None
        gamma_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "চলুন।\nঅথবা,\nআয়।\n(প্রসঙ্গ অনুযায়ী উপযুক্ত অনুবাদ ব্যবহার করা উচিত।)"
                        )
                    }
                }
            ]
        }
        qwen_response = MagicMock()
        qwen_response.raise_for_status.return_value = None
        qwen_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            'চলুন।\n###TRANSLATION_END###{"translation_quality": 0.93}'
                        )
                    }
                }
            ]
        }

        with patch.object(
            service._http_client,
            "post",
            new=AsyncMock(side_effect=[gamma_response, qwen_response]),
        ) as mock_post:
            result = await service._real_translate(
                "Come on.",
                "en",
                "bn",
                custom_translation_routing=True,
            )

        assert result.text == "চলুন।"
        assert result.translation_quality == 0.93
        assert mock_post.await_count == 2
        assert (
            mock_post.await_args_list[0]
            .args[0]
            .startswith("https://gamma.example.com/")
        )
        assert (
            mock_post.await_args_list[1].args[0].startswith("https://qwen.example.com/")
        )

    def test_endpoint_model_placeholder_uses_safe_replacement(self):
        """Test endpoint model substitution only replaces the documented token."""
        config = self._service_config()
        config["providers"]["translategamma"][
            "endpoint"
        ] = "/models/{model}:translate/{extra}"
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        url, _, _ = service._build_translation_request("Hello", "en", "gu")

        assert (
            url == "https://gamma.example.com/models/translategamma:translate/{extra}"
        )

    def test_legacy_temperature_is_coerced_to_float(self):
        """Test flat translation temperature config is normalized at startup."""
        config = self._service_config()
        config.pop("default_provider")
        config["routing"] = []
        config["temperature"] = "0.25"
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        _, _, payload = service._build_translation_request("Bonjour", "fr", "de")

        assert service.temperature == 0.25
        assert payload["temperature"] == 0.25

    def test_anthropic_version_unresolved_env_falls_back_to_default(self):
        """Test unresolved Anthropic version placeholders use the safe default."""
        config = self._service_config()
        config["providers"]["qwen3"].update(
            {
                "api_format": "anthropic_messages",
                "endpoint": "/v1/messages",
                "anthropic_version": "${ANTHROPIC_VERSION}",
            }
        )
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        _, headers, _ = service._build_translation_request("Bonjour", "fr", "de")

        assert headers["anthropic-version"] == "2023-06-01"

    def test_routing_falls_back_to_default_provider(self):
        """Test unmatched language pairs use default_provider."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        url, headers, payload = service._build_translation_request(
            "Bonjour", "fr", "de"
        )

        assert url == "https://qwen.example.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer qwen-key"
        assert payload["model"] == "qwen3"

    def test_routing_matches_base_language_codes(self):
        """Test routes match regional language variants by base code."""
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        _, _, payload = service._build_translation_request("Hello", "en-US", "gu-IN")

        assert payload["model"] == "translategamma"

    def test_routing_unknown_provider_fails_clearly(self):
        """Test invalid routing provider names fail before the API call."""
        config = self._service_config()
        config["routing"] = [
            {"priority": 1, "target_langs": ["gu"], "provider": "missing"}
        ]
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        with pytest.raises(ValueError, match="Unknown translation provider 'missing'"):
            service._build_translation_request("Hello", "en", "gu")

    def test_routing_rule_missing_provider_key_raises(self):
        """Test matching routing rules must name a provider."""
        config = self._service_config()
        config["routing"] = [{"priority": 1, "target_langs": ["gu"]}]
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        with pytest.raises(
            ValueError, match="Translation routing rule is missing provider"
        ):
            service._build_translation_request("Hello", "en", "gu")

    def test_routing_wildcard_language_matches_anything(self):
        """Test wildcard language rules match any requested language."""
        config = self._service_config()
        config["routing"] = [
            {"priority": 5, "target_langs": ["*"], "provider": "translategamma"}
        ]
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        url, _, payload = service._build_translation_request("Hello", "en", "zh")

        assert url == "https://gamma.example.com/v1/chat/completions"
        assert payload["model"] == "translategamma"

    def test_invalid_routing_rule_logs_warning(self, caplog):
        """Test malformed routing entries are visible in logs."""
        config = self._service_config()
        config["routing"] = [
            "not-a-rule",
            {"priority": 5, "target_langs": ["zh"], "provider": "translategamma"},
        ]
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        with caplog.at_level("WARNING"):
            _, _, payload = service._build_translation_request("Hello", "en", "zh")

        assert "Skipping invalid translation routing rule" in caplog.text
        assert "index=0" in caplog.text
        assert "type=str" in caplog.text
        assert payload["model"] == "translategamma"

    def test_empty_routing_rule_logs_warning_and_falls_back(self, caplog):
        """Test empty rule mappings are not treated as catch-all rules."""
        config = self._service_config()
        config["routing"] = [{}]
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        with caplog.at_level("WARNING"):
            _, _, payload = service._build_translation_request("Hello", "en", "gu")

        assert "Skipping empty translation routing rule" in caplog.text
        assert "index=0" in caplog.text
        assert payload["model"] == "qwen3"

    def test_invalid_language_matcher_logs_warning_and_does_not_match(self, caplog):
        """Test malformed language matcher values are not treated as wildcards."""
        config = self._service_config()
        config["routing"] = [
            {"priority": 1, "target_langs": 123, "provider": "translategamma"}
        ]
        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = config
            service = TranslationService()

        with caplog.at_level("WARNING"):
            _, _, payload = service._build_translation_request("Hello", "en", "gu")

        assert "Skipping invalid translation routing language matcher" in caplog.text
        assert "field=target_langs" in caplog.text
        assert "type=int" in caplog.text
        assert payload["model"] == "qwen3"

    @pytest.mark.asyncio
    async def test_real_translate_uses_routed_provider(self):
        """Test real translation sends the request to the routed provider."""
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "નમસ્તે"}}]}

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        with patch.object(
            service._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = response
            result = await service._real_translate("Hello", "en", "gu")

        assert result.success is True
        assert result.text == "નમસ્તે"
        call_args = mock_post.call_args
        assert call_args.args[0] == "https://gamma.example.com/v1/chat/completions"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer gamma-key"
        assert call_args.kwargs["json"]["model"] == "translategamma"

    @pytest.mark.asyncio
    async def test_public_translate_flag_switches_provider_routing(self):
        """The public translation API defaults to legacy and opts into routing explicitly."""
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "નમસ્તે"}}]}

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        with patch.object(
            service._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = response
            legacy_result = await service.translate("Hello", "en", "gu")
            custom_result = await service.translate(
                "Hello", "en", "gu", custom_translation_routing=True
            )

        assert legacy_result.success is True
        assert custom_result.success is True
        assert (
            mock_post.call_args_list[0].args[0]
            == "https://qwen.example.com/v1/chat/completions"
        )
        assert (
            mock_post.call_args_list[1].args[0]
            == "https://gamma.example.com/v1/chat/completions"
        )

    @pytest.mark.asyncio
    async def test_translate_real_mode_forwards_custom_instruction(self):
        """Test custom instructions survive the public real-translate path."""
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "નમસ્તે"}}]}

        with patch("app.services.translation_service.get_config") as mock_config:
            mock_config.return_value.translation = self._service_config()
            service = TranslationService()

        with patch.object(
            service._http_client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = response
            result = await service.translate(
                "Hello",
                "en",
                "gu",
                custom_instruction="Keep product names in English.",
            )

        assert result.success is True
        messages = mock_post.call_args.kwargs["json"]["messages"]
        assert "User translation instruction (high priority)" in messages[0]["content"]
        assert "Keep product names in English." in messages[0]["content"]
        assert "Keep product names in English." not in messages[1]["content"]


class TestVisualContextService:
    """Test visual context request construction."""

    def test_openai_chat_visual_context_request_uses_image_url(self):
        """Test OpenAI chat vision payload includes the screenshot data URL."""
        from unittest.mock import patch

        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "endpoint": "/v1/chat/completions",
                "api_format": "openai_chat",
                "api_key": "test-key",
                "model": "gpt-4o-mini",
            }
            mock_config.return_value.translation = {}

            service = VisualContextService()
            image_url = "data:image/jpeg;base64,aGVsbG8="

            url, headers, payload = service._build_request(
                image_url,
                "Summarize this screenshot.",
            )

            assert url == "https://api.openai.com/v1/chat/completions"
            assert headers["Authorization"] == "Bearer test-key"
            content = payload["messages"][1]["content"]
            assert content[0]["type"] == "text"
            assert content[1]["image_url"]["url"] == image_url

    def test_openai_chat_visual_context_disables_thinking_by_default(self):
        """Test OpenAI chat visual context payload requests no thinking mode."""
        from unittest.mock import patch

        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "endpoint": "/v1/chat/completions",
                "api_format": "openai_chat",
                "api_key": "test-key",
                "model": "coding-ninja",
            }
            mock_config.return_value.translation = {}

            service = VisualContextService()
            _, _, payload = service._build_request(
                "data:image/jpeg;base64,aGVsbG8=",
                "Summarize this screenshot.",
            )

        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_openai_chat_visual_context_null_content_returns_empty(self):
        """Test OpenAI chat visual context parser tolerates null content."""
        from unittest.mock import patch

        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": False,
                "api_format": "openai_chat",
            }
            mock_config.return_value.translation = {}

            service = VisualContextService()
            result = service._parse_response(
                {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
            )

        assert result == ""

    def test_openai_chat_visual_context_list_content_extracts_text(self):
        """Test OpenAI chat visual context parser handles list content."""
        from unittest.mock import patch

        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": False,
                "api_format": "openai_chat",
            }
            mock_config.return_value.translation = {}

            service = VisualContextService()
            result = service._parse_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": " Shared "},
                                    {"type": "text", "text": " context "},
                                ]
                            }
                        }
                    ]
                }
            )

        assert result == "Shared  context"

    def test_visual_context_temperature_uses_config_value_fallback(self):
        """Test visual context temperature ignores unresolved env placeholders."""
        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": False,
                "temperature": "${VISUAL_CONTEXT_TEMPERATURE}",
            }
            mock_config.return_value.translation = {}

            service = VisualContextService()

        assert service.temperature == 0.0

    @pytest.mark.asyncio
    async def test_visual_context_retries_transient_api_failure(self):
        """Test visual context retries transient provider failures."""
        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": False,
                "base_url": "https://api.openai.com",
                "endpoint": "/v1/chat/completions",
                "api_format": "openai_chat",
                "api_key": "test-key",
                "model": "gpt-4o-mini",
            }
            mock_config.return_value.translation = {}

            service = VisualContextService()
            response = MagicMock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [{"message": {"content": "Shared context"}}]
            }

            with (
                patch.object(
                    service._http_client,
                    "post",
                    new_callable=AsyncMock,
                    side_effect=[httpx.HTTPError("temporary"), response],
                ) as mock_post,
                patch(
                    "app.services.visual_context_service.asyncio.sleep",
                    new_callable=AsyncMock,
                ) as mock_sleep,
            ):
                result = await service.summarize_screenshot(
                    "data:image/jpeg;base64,aGVsbG8=", "de", "en"
                )

        assert result == "Shared context"
        assert mock_post.call_count == 2
        mock_sleep.assert_awaited_once_with(0.5)

    def test_visual_context_summary_is_trimmed(self):
        """Test visual context summaries are whitespace-normalized and bounded."""
        from unittest.mock import patch

        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": True,
                "max_summary_chars": 10,
            }
            mock_config.return_value.translation = {"mock_mode": True}

            service = VisualContextService()

            assert service._trim_summary("  alpha   beta   gamma  ") == "alpha beta"

    def test_visual_context_mock_mode_logs_warning_when_enabled(self, caplog):
        """Test enabled visual mock mode logs a clear warning."""
        from unittest.mock import patch

        from app.services.visual_context_service import VisualContextService

        with patch("app.services.visual_context_service.get_config") as mock_config:
            mock_config.return_value.visual_context = {
                "enabled": True,
                "mock_mode": True,
            }
            mock_config.return_value.translation = {"mock_mode": True}

            with caplog.at_level("WARNING"):
                VisualContextService()

        assert "VisualContextService is in mock mode" in caplog.text
