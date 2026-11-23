
from typing import Literal

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    name: str

    category: Literal[
        "programming_language",
        "experience",
        "architecture",
        "distributed_systems",
        "education",
        "machine_learning",
        "cloud",
        "other"
    ]

    priority: Literal["mandatory", "preferred"]

    evidence_signals: list[str] = Field(default_factory=list)

    alternatives: list[str] = Field(default_factory=list)

    match_mode: Literal["all", "any"] = "all"

    minimum_years: float | None = None


class JobProfile(BaseModel):
    role_title: str
    role_scope: str

    mandatory_requirements: list[Requirement]
    preferred_requirements: list[Requirement]

    responsibilities: list[str] = Field(default_factory=list)

    role_specializations: list[str] = Field(default_factory=list)

    clarification_notes: list[str] = Field(default_factory=list)
