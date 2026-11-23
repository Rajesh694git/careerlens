
import httpx
from pydantic import BaseModel

from app.schemas.analysis import AnalysisResponse


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

    async with httpx.AsyncClient(timeout=240.0) as client:
        response = await client.post(
            OLLAMA_URL,
            json=payload,
        )
        response.raise_for_status()

    result = response.json()
    print("Ollama done reason:", result.get("done_reason"))
    print("Ollama generated tokens:", result.get("eval_count"))
    print("Ollama message:", result.get("message"))
    message = result.get("message") or {}
    content = (message.get("content") or "").strip()

    if not content:
        raise RuntimeError(
            "Ollama returned empty content. "
            f"Done reason: {result.get('done_reason')}, "
            f"Generated tokens: {result.get('eval_count')}, "
            f"Thinking tokens: "
            f"{len(message.get('thinking') or '')}"
        )

    try:
        return response_schema.model_validate_json(content)
    except Exception as exc:
        raise ValueError(
            f"Invalid structured response from Ollama: {exc}",
        ) from exc


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