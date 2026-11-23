
from typing import Literal

from pydantic import BaseModel, Field


class Skill(BaseModel):
    name: str
    category: Literal[
        "programming_language",
        "framework",
        "database",
        "cloud",
        "architecture",
        "machine_learning",
        "devops",
        "testing",
        "other"
    ]


class WorkExperience(BaseModel):
    company: str
    role: str
    start_date: str | None = None
    end_date: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: str
    institution: str
    start_year: str | None = None
    end_year: str | None = None
    specialization: str | None = None


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    professional_summary: str
    skills: list[Skill] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    additional_evidence: list[str] = Field(default_factory=list)
    clarification_notes: list[str] = Field(default_factory=list)
