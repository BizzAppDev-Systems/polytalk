# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for Pydantic schemas.
"""

from app.schemas.translation import (
    ErrorResponse,
    HealthResponse,
    TextTranslationRequest,
)
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


class TestTextTranslationRequestGlossaryTerms:
    """glossary_terms normalizes blanks and enforces the count bound."""

    def test_glossary_terms_trimmed(self):
        req = TextTranslationRequest(
            text="hello",
            source_language="en",
            target_language="de",
            glossary_terms=["  transformer  ", "attention"],
        )
        assert req.glossary_terms == ["transformer", "attention"]

    def test_glossary_terms_blank_only_normalizes_to_none(self):
        req = TextTranslationRequest(
            text="hello",
            source_language="en",
            target_language="de",
            glossary_terms=["", "   "],
        )
        assert req.glossary_terms is None

    def test_glossary_terms_omitted_defaults_to_none(self):
        req = TextTranslationRequest(
            text="hello", source_language="en", target_language="de"
        )
        assert req.glossary_terms is None

    def test_glossary_terms_over_500_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TextTranslationRequest(
                text="hello",
                source_language="en",
                target_language="de",
                glossary_terms=[f"term{i}" for i in range(501)],
            )
