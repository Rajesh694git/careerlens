
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    resume_text: str = Field(min_length=50)
    job_description: str = Field(min_length=50)


class AnalysisResponse(BaseModel):
    match_score: int = Field(ge=0, le=100)
    summary: str
    matched_skills: list[str]
    missing_skills: list[str]
    strengths: list[str]
    recommendations: list[str]
    caveats: list[str] = []
