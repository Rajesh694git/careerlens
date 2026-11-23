
from fastapi import APIRouter

from app.schemas.matching import MatchingRequest, MatchingResponse
from app.services.matching_service import match_candidate

router = APIRouter(
    prefix="/api/v1/matching",
    tags=["Candidate Matching"]
)


@router.post("/compare", response_model=MatchingResponse)
async def compare_candidate(request: MatchingRequest):

    return match_candidate(
        candidate=request.candidate_profile,
        job=request.job_profile
    )
