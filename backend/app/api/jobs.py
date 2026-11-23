
from fastapi import APIRouter, Form, HTTPException

from app.schemas.job import JobProfile
from app.services.job_service import extract_job_profile


router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["Job Analysis"]
)


@router.post("/extract", response_model=JobProfile)
async def extract_job_requirements(
    job_description: str = Form(...)
):

    if len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Job description is too short."
        )

    if len(job_description) > 20000:
        raise HTTPException(
            status_code=413,
            detail="Job description is too large."
        )

    try:
        return await extract_job_profile(job_description)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc)
        )
