
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobProfile


class RequirementResult(BaseModel):
    requirement_name: str
    category: str
    priority: Literal["mandatory", "preferred"]

    status: Literal[
        "matched",
        "partially_matched",
        "not_evidenced"
    ]

    score: float = Field(ge=0, le=100)

    matched_evidence: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    explanation: str


class MatchingResponse(BaseModel):
    match_score: float = Field(ge=0, le=100)

    mandatory_score: float = Field(ge=0, le=100)
    preferred_score: float = Field(ge=0, le=100)

    total_experience_years: float

    matched_requirements: list[RequirementResult]
    partially_matched_requirements: list[RequirementResult]
    missing_requirements: list[RequirementResult]

    strengths: list[str]
    skill_gaps: list[str]
    recommendations: list[str]

    disclaimer: str


class MatchingRequest(BaseModel):
    candidate_profile: CandidateProfile
    job_profile: JobProfile

