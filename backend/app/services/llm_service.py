import logging
import time

import httpx
from pydantic import BaseModel, ValidationError

from app.schemas.analysis import AnalysisResponse


logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:4b"


async def generate_structured(
    prompt: str,
    response_schema: type[BaseModel],
) -> BaseModel:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "format": response_schema.model_json_schema(),
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 2500,
        },
    }

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json=payload,
            )
            response.raise_for_status()

    except httpx.HTTPError as exc:
        logger.error(
            "Ollama request failed: model=%s error_type=%s",
            MODEL_NAME,
            type(exc).__name__,
        )
        raise

    result = response.json()
    message = result.get("message") or {}
    content = (message.get("content") or "").strip()

    duration_seconds = time.perf_counter() - start_time

    logger.info(
        "Ollama generation completed: model=%s duration_seconds=%.2f "
        "done_reason=%s prompt_tokens=%s generated_tokens=%s",
        MODEL_NAME,
        duration_seconds,
        result.get("done_reason"),
        result.get("prompt_eval_count"),
        result.get("eval_count"),
    )

    if not content:
        logger.warning(
            "Ollama returned empty content: model=%s done_reason=%s "
            "generated_tokens=%s thinking_tokens=%s",
            MODEL_NAME,
            result.get("done_reason"),
            result.get("eval_count"),
            len(message.get("thinking") or ""),
        )
        raise RuntimeError("Ollama returned empty content.")

    try:
        return response_schema.model_validate_json(content)

    except ValidationError as exc:
        logger.warning(
            "Ollama response validation failed: model=%s error_type=%s",
            MODEL_NAME,
            type(exc).__name__,
        )
        raise ValueError(
            "Ollama returned a response that does not match the expected schema."
        ) from None


async def analyse_resume(
    resume_text: str,
    job_description: str,
) -> AnalysisResponse:
    prompt = f"""
Compare the candidate's resume against the job description.

Use only evidence from the supplied documents.
Do not invent experience or qualifications.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Provide matched skills, missing skills, strengths,
recommendations and caveats.
"""

    return await generate_structured(
        prompt,
        AnalysisResponse,
    )
