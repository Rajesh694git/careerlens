from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

import httpx

from app.constants import MAX_RESUME_FILE_SIZE_BYTES
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.document_parser import extract_resume_text
from app.services.llm_service import analyse_resume

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

MAX_JD_LENGTH = 20000


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    try:
        return await analyse_resume(
            resume_text=request.resume_text,
            job_description=request.job_description,
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Local Ollama service is unavailable.",
        )
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Error communicating with Ollama.",
        )
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="AI returned an invalid analysis response.",
        )


@router.post("/analyze-upload", response_model=AnalysisResponse)
async def analyze_upload(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    if not resume.filename:
        raise HTTPException(400, "Filename is required.")

    if len(job_description.strip()) < 50:
        raise HTTPException(422, "Job description is too short.")

    if len(job_description) > MAX_JD_LENGTH:
        raise HTTPException(413, "Job description is too large.")

    try:
        content = await resume.read(MAX_RESUME_FILE_SIZE_BYTES + 1)

        if len(content) > MAX_RESUME_FILE_SIZE_BYTES:
            raise HTTPException(413, "Maximum resume size is 5 MB.")

        resume_text = await run_in_threadpool(
            extract_resume_text,
            resume.filename,
            content,
        )

        result = await analyse_resume(
            resume_text=resume_text,
            job_description=job_description,
        )

        return result

    except ValueError as exc:
        raise HTTPException(422, str(exc))

    except httpx.ConnectError:
        raise HTTPException(
            503,
            "Local Ollama service is unavailable.",
        )

    except httpx.HTTPError:
        raise HTTPException(
            502,
            "Error communicating with Ollama.",
        )

    finally:
        await resume.close()
