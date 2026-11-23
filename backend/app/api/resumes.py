from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.constants import MAX_RESUME_FILE_SIZE_BYTES
from app.schemas.candidate import CandidateProfile
from app.services.candidate_service import extract_candidate_profile
from app.services.document_parser import extract_resume_text

router = APIRouter(
    prefix="/api/v1/resumes",
    tags=["Resume Analysis"],
)


@router.post("/extract", response_model=CandidateProfile)
async def extract_resume(
    resume: UploadFile = File(...),
):
    try:
        if not resume.filename:
            raise HTTPException(
                status_code=400,
                detail="Resume filename is missing.",
            )

        content = await resume.read(MAX_RESUME_FILE_SIZE_BYTES + 1)

        if len(content) > MAX_RESUME_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Resume exceeds the 5 MB limit.",
            )

        resume_text = await run_in_threadpool(
            extract_resume_text,
            resume.filename,
            content,
        )

        return await extract_candidate_profile(resume_text)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    finally:
        await resume.close()
