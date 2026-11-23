import asyncio
import json
import logging

import httpx
import pytest

from app.schemas.analysis import AnalysisResponse
from app.services import llm_service


def valid_analysis():
    return {
        "match_score": 82,
        "summary": "Generated response containing OUTPUT_SECRET",
        "matched_skills": ["Python"],
        "missing_skills": ["Kubernetes"],
        "strengths": ["Backend experience"],
        "recommendations": ["Improve cloud knowledge"],
        "caveats": [],
    }


def install_mock_ollama(monkeypatch, handler):
    original_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def create_client(*, timeout):
        return original_async_client(
            transport=transport,
            timeout=timeout,
        )

    monkeypatch.setattr(
        llm_service.httpx,
        "AsyncClient",
        create_client,
    )


def test_generate_structured_success(monkeypatch, caplog):
    prompt = "RESUME_SECRET candidate information"

    def handler(request):
        payload = json.loads(request.content)

        assert payload["model"] == llm_service.MODEL_NAME
        assert payload["messages"][0]["content"] == prompt
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0

        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(valid_analysis())
                },
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 200,
            },
            request=request,
        )

    install_mock_ollama(monkeypatch, handler)

    caplog.set_level(logging.INFO, logger=llm_service.__name__)

    result = asyncio.run(
        llm_service.generate_structured(
            prompt,
            AnalysisResponse,
        )
    )

    assert isinstance(result, AnalysisResponse)
    assert result.match_score == 82
    assert result.matched_skills == ["Python"]

    # Sensitive prompt and generated content must not appear in logs.
    assert "RESUME_SECRET" not in caplog.text
    assert "OUTPUT_SECRET" not in caplog.text


def test_generate_structured_empty_response(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "thinking": "PRIVATE_THINKING",
                },
                "done_reason": "stop",
                "eval_count": 0,
            },
            request=request,
        )

    install_mock_ollama(monkeypatch, handler)

    with pytest.raises(RuntimeError, match="empty content"):
        asyncio.run(
            llm_service.generate_structured(
                "test prompt",
                AnalysisResponse,
            )
        )


def test_generate_structured_invalid_json(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "{invalid-json"
                }
            },
            request=request,
        )

    install_mock_ollama(monkeypatch, handler)

    with pytest.raises(ValueError, match="does not match the expected schema"):
        asyncio.run(
            llm_service.generate_structured(
                "test prompt",
                AnalysisResponse,
            )
        )


def test_generate_structured_schema_validation_failure(monkeypatch):
    invalid_analysis = valid_analysis()
    invalid_analysis["match_score"] = 150

    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(invalid_analysis)
                }
            },
            request=request,
        )

    install_mock_ollama(monkeypatch, handler)

    with pytest.raises(ValueError, match="does not match the expected schema"):
        asyncio.run(
            llm_service.generate_structured(
                "test prompt",
                AnalysisResponse,
            )
        )


def test_generate_structured_http_error_does_not_log_response(
    monkeypatch,
    caplog,
):
    def handler(request):
        return httpx.Response(
            503,
            text="SERVER_SECRET internal details",
            request=request,
        )

    install_mock_ollama(monkeypatch, handler)

    caplog.set_level(logging.ERROR, logger=llm_service.__name__)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            llm_service.generate_structured(
                "RESUME_SECRET",
                AnalysisResponse,
            )
        )

    assert "SERVER_SECRET" not in caplog.text
    assert "RESUME_SECRET" not in caplog.text


def test_generate_structured_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout(
            "Ollama request timed out",
            request=request,
        )

    install_mock_ollama(monkeypatch, handler)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(
            llm_service.generate_structured(
                "test prompt",
                AnalysisResponse,
            )
        )
