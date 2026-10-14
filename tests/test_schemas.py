# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for Pydantic schemas.
"""

import pytest

from app.schemas.translation import HealthResponse, ErrorResponse
from app.version import __version__


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_health_response_minimal(self):
        """Test HealthResponse with minimal data."""
        response = HealthResponse(status="healthy")
        assert response.status == "healthy"
        assert response.version == __version__

    def test_health_response_full(self):
        """Test HealthResponse with all fields."""
        response = HealthResponse(status="healthy", version="2.0.0")
        assert response.status == "healthy"
        assert response.version == "2.0.0"

    def test_health_response_different_statuses(self):
        """Test HealthResponse with different status values."""
        response_unhealthy = HealthResponse(status="unhealthy")
        assert response_unhealthy.status == "unhealthy"

        response_degraded = HealthResponse(status="degraded")
        assert response_degraded.status == "degraded"

    def test_health_response_model_dump(self):
        """Test HealthResponse model_dump method."""
        response = HealthResponse(status="healthy", version=__version__)
        data = response.model_dump()
        assert data["status"] == "healthy"
        assert data["version"] == __version__

    def test_health_response_model_dump_json(self):
        """Test HealthResponse model_dump_json method."""
        response = HealthResponse(status="healthy", version=__version__)
        json_str = response.model_dump_json()
        assert "healthy" in json_str
        assert __version__ in json_str

    def test_health_response_from_dict(self):
        """Test HealthResponse from dictionary."""
        data = {"status": "healthy", "version": __version__}
        response = HealthResponse(**data)
        assert response.status == "healthy"
        assert response.version == __version__


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_response_minimal(self):
        """Test ErrorResponse with minimal data."""
        response = ErrorResponse(error="Something went wrong")
        assert response.success is False
        assert response.error == "Something went wrong"
        assert response.details is None

    def test_error_response_full(self):
        """Test ErrorResponse with all fields."""
        response = ErrorResponse(
            error="Error message", success=False, details="Detailed info"
        )
        assert response.success is False
        assert response.error == "Error message"
        assert response.details == "Detailed info"

    def test_error_response_with_details(self):
        """Test ErrorResponse with details field."""
        response = ErrorResponse(error="Error", details="Additional context")
        assert response.details == "Additional context"

    def test_error_response_without_details(self):
        """Test ErrorResponse without details field."""
        response = ErrorResponse(error="Error")
        assert response.details is None

    def test_error_response_model_dump(self):
        """Test ErrorResponse model_dump method."""
        response = ErrorResponse(error="Error", details="Details")
        data = response.model_dump()
        assert data["success"] is False
        assert data["error"] == "Error"
        assert data["details"] == "Details"

    def test_error_response_model_dump_json(self):
        """Test ErrorResponse model_dump_json method."""
        response = ErrorResponse(error="Error", details="Details")
        json_str = response.model_dump_json()
        assert "Error" in json_str
        assert "Details" in json_str

    def test_error_response_from_dict(self):
        """Test ErrorResponse from dictionary."""
        data = {"error": "Test error", "details": "Test details"}
        response = ErrorResponse(**data)
        assert response.error == "Test error"
        assert response.details == "Test details"

    def test_error_response_empty_string_error(self):
        """Test ErrorResponse with empty string error."""
        response = ErrorResponse(error="")
        assert response.error == ""
        assert response.success is False


class TestTextTranslationRequest:
    """Tests for TextTranslationRequest schema."""

    def test_valid_translation_request(self):
        """Test with valid fields."""
        from app.schemas.translation import TextTranslationRequest

        request = TextTranslationRequest(
            text="કેમ છો?",
            source_language="gu",
            target_language="en",
        )
        assert request.text == "કેમ છો?"
        assert request.source_language == "gu"
        assert request.target_language == "en"
        assert request.custom_instruction is None

    def test_translation_request_with_custom_instruction(self):
        """Test with custom instruction."""
        from app.schemas.translation import TextTranslationRequest

        request = TextTranslationRequest(
            text="Hello",
            source_language="en",
            target_language="gu",
            custom_instruction="Keep it formal",
        )
        assert request.custom_instruction == "Keep it formal"

    def test_translation_request_max_length(self):
        """Test 5000 character text is valid."""
        from app.schemas.translation import TextTranslationRequest

        request = TextTranslationRequest(
            text="A" * 5000,
            source_language="en",
            target_language="gu",
        )
        assert len(request.text) == 5000

    def test_translation_request_empty_text_rejected(self):
        """Test empty text is rejected."""
        from app.schemas.translation import TextTranslationRequest

        with pytest.raises(Exception):
            TextTranslationRequest(
                text="",
                source_language="en",
                target_language="gu",
            )

    def test_translation_request_too_long_text_rejected(self):
        """Test text > 5000 chars is rejected."""
        from app.schemas.translation import TextTranslationRequest

        with pytest.raises(Exception):
            TextTranslationRequest(
                text="A" * 5001,
                source_language="en",
                target_language="gu",
            )

    def test_translation_request_invalid_language_code(self):
        """Test invalid language code is rejected."""
        from app.schemas.translation import TextTranslationRequest

        with pytest.raises(Exception):
            TextTranslationRequest(
                text="Hello",
                source_language="invalid",
                target_language="en",
            )

    def test_translation_request_valid_3letter_code(self):
        """Test 3-letter language code is valid."""
        from app.schemas.translation import TextTranslationRequest

        request = TextTranslationRequest(
            text="Hello",
            source_language="eng",
            target_language="guj",
        )
        assert request.source_language == "eng"
        assert request.target_language == "guj"

    def test_translation_request_with_region_code(self):
        """Test language code with region is valid."""
        from app.schemas.translation import TextTranslationRequest

        request = TextTranslationRequest(
            text="Hello",
            source_language="en_US",
            target_language="es_MX",
        )
        assert request.source_language == "en_US"
        assert request.target_language == "es_MX"

    def test_translation_request_custom_instruction_max_length(self):
        """Test custom instruction max 1000 chars."""
        from app.schemas.translation import TextTranslationRequest

        request = TextTranslationRequest(
            text="Hello",
            source_language="en",
            target_language="gu",
            custom_instruction="A" * 1000,
        )
        assert len(request.custom_instruction) == 1000

    def test_translation_request_custom_instruction_too_long(self):
        """Test custom instruction > 1000 chars is rejected."""
        from app.schemas.translation import TextTranslationRequest

        with pytest.raises(Exception):
            TextTranslationRequest(
                text="Hello",
                source_language="en",
                target_language="gu",
                custom_instruction="A" * 1001,
            )

    def test_translation_request_model_dump(self):
        """Test model_dump method."""
        from app.schemas.translation import TextTranslationRequest

        request = TextTranslationRequest(
            text="Hello",
            source_language="en",
            target_language="gu",
        )
        data = request.model_dump()
        assert data["text"] == "Hello"
        assert data["source_language"] == "en"
        assert data["target_language"] == "gu"
        assert data["custom_instruction"] is None

    def test_translation_request_from_dict(self):
        """Test creating from dictionary."""
        from app.schemas.translation import TextTranslationRequest

        data = {
            "text": "Test",
            "source_language": "en",
            "target_language": "gu",
        }
        request = TextTranslationRequest(**data)
        assert request.text == "Test"


class TestTextTranslationResponse:
    """Tests for TextTranslationResponse schema."""

    def test_valid_response(self):
        """Test with valid fields."""
        from app.schemas.translation import TextTranslationResponse

        response = TextTranslationResponse(
            translated_text="કેમ છો?",
            source_language="gu",
            target_language="en",
        )
        assert response.translated_text == "કેમ છો?"
        assert response.source_language == "gu"
        assert response.target_language == "en"

    def test_response_model_dump(self):
        """Test model_dump method."""
        from app.schemas.translation import TextTranslationResponse

        response = TextTranslationResponse(
            translated_text="Hello",
            source_language="gu",
            target_language="en",
        )
        data = response.model_dump()
        assert data["translated_text"] == "Hello"

    def test_response_model_dump_json(self):
        """Test model_dump_json method."""
        from app.schemas.translation import TextTranslationResponse

        response = TextTranslationResponse(
            translated_text="Hello",
            source_language="gu",
            target_language="en",
        )
        json_str = response.model_dump_json()
        assert "Hello" in json_str

    def test_response_from_dict(self):
        """Test creating from dictionary."""
        from app.schemas.translation import TextTranslationResponse

        data = {
            "translated_text": "Hello",
            "source_language": "gu",
            "target_language": "en",
        }
        response = TextTranslationResponse(**data)
        assert response.translated_text == "Hello"


class TestTextSpeechRequest:
    """Tests for TextSpeechRequest schema."""

    def test_valid_speech_request_no_voice(self):
        """Test with valid fields, no voice."""
        from app.schemas.translation import TextSpeechRequest

        request = TextSpeechRequest(
            text="કેમ છો?",
            language="gu",
        )
        assert request.text == "કેમ છો?"
        assert request.language == "gu"
        assert request.voice is None

    def test_speech_request_with_voice(self):
        """Test with voice parameter."""
        from app.schemas.translation import TextSpeechRequest

        request = TextSpeechRequest(
            text="કેમ છો?",
            language="gu",
            voice="gu_uk_ritesh_dabhi-medium",
        )
        assert request.voice == "gu_uk_ritesh_dabhi-medium"

    def test_speech_request_empty_text_rejected(self):
        """Test empty text is rejected."""
        from app.schemas.translation import TextSpeechRequest

        with pytest.raises(Exception):
            TextSpeechRequest(
                text="",
                language="gu",
            )

    def test_speech_request_too_long_text_rejected(self):
        """Test text > 5000 chars is rejected."""
        from app.schemas.translation import TextSpeechRequest

        with pytest.raises(Exception):
            TextSpeechRequest(
                text="A" * 5001,
                language="gu",
            )

    def test_speech_request_invalid_language(self):
        """Test invalid language code is rejected."""
        from app.schemas.translation import TextSpeechRequest

        with pytest.raises(Exception):
            TextSpeechRequest(
                text="Hello",
                language="invalid",
            )

    def test_speech_request_invalid_voice_empty(self):
        """Test empty voice is rejected."""
        from app.schemas.translation import TextSpeechRequest

        with pytest.raises(Exception):
            TextSpeechRequest(
                text="Hello",
                language="en",
                voice="",
            )

    def test_speech_request_model_dump(self):
        """Test model_dump method."""
        from app.schemas.translation import TextSpeechRequest

        request = TextSpeechRequest(
            text="Hello",
            language="en",
            voice="en_US-lessac-medium",
        )
        data = request.model_dump()
        assert data["text"] == "Hello"
        assert data["voice"] == "en_US-lessac-medium"
