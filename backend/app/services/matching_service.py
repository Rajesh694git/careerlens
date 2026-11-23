import re
from datetime import datetime

from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobProfile, Requirement
from app.schemas.matching import MatchingResponse, RequirementResult


# Canonicalise common technology naming variations.
ALIASES = {
    "golang": "go",
    "fast api": "fastapi",
    "fastapi": "fastapi",
    "pytorch": "pytorch",
    "springboot": "spring boot",
    "spring boot": "spring boot",
    "rest apis": "rest api",
    "restful api": "rest api",
    "junit": "junit",
    "jacoco": "jacoco",
}


def normalise(value: str) -> str:
    value = value.lower().strip()
    value = ALIASES.get(value, value)
    value = re.sub(r"[^a-z0-9+#]+", " ", value)
    return " ".join(value.split())


def contains_term(text: str, term: str) -> bool:
    """Check for a complete phrase rather than a loose substring."""
    text = normalise(text)
    term = normalise(term)

    if not text or not term:
        return False

    return f" {term} " in f" {text} "


def is_bachelors_degree(degree: str) -> bool:
    """Recognise common bachelor's-level degree names and abbreviations."""
    normalized = normalise(degree)
    patterns = (
        r"\bb\s*e\b",         # B.E. / B E / BE after punctuation normalisation
        r"\bb\s*tech\b",      # B.Tech / B Tech
        r"\bbachelor\b",       # Bachelor of Engineering, etc.
        r"\bb\s*sc\b",        # B.Sc. / B Sc
        r"\bbs\b",              # BS
        r"\bb\s*ca\b",        # BCA / B CA
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def education_matches(
    candidate: CandidateProfile,
    alternative: str,
) -> bool:
    """Match education against academic credentials, not work experience."""
    alternative_normalized = normalise(alternative)

    # Employment duration alone is not proof of an equivalent qualification.
    if "equivalent experience" in alternative_normalized:
        return False

    if "bachelor" in alternative_normalized:
        return any(
            is_bachelors_degree(item.degree)
            for item in candidate.education
        )

    return any(
        contains_term(item.degree, alternative)
        for item in candidate.education
    )


def employment_month(date_text: str | None, is_end: bool = False):
    """Convert dates such as 'Dec 2023' into a month index."""
    if not date_text:
        return None

    if date_text.strip().lower() in {"present", "current", "ongoing"}:
        now = datetime.now()
        return now.year * 12 + now.month - 1

    formats = ["%b %Y", "%B %Y", "%Y"]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_text.strip(), fmt)
            month = parsed.month if fmt != "%Y" else (12 if is_end else 1)
            return parsed.year * 12 + month - 1
        except ValueError:
            continue

    return None


def calculate_months(intervals: list[tuple[int, int]]) -> int:
    """Merge overlapping employment periods to avoid double counting."""
    if not intervals:
        return 0

    intervals = sorted(intervals)
    total = 0
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end

    total += current_end - current_start + 1
    return total


def employment_intervals(candidate: CandidateProfile) -> list[tuple[int, int]]:
    intervals = []

    for job in candidate.work_experience:
        start = employment_month(job.start_date)
        end = employment_month(job.end_date, is_end=True)

        if start is not None and end is not None and end >= start:
            intervals.append((start, end))

    return intervals


def total_experience_years(candidate: CandidateProfile) -> float:
    months = calculate_months(employment_intervals(candidate))
    return round(months / 12, 1)


def job_evidence_text(job) -> str:
    return " ".join([
        job.company or "",
        job.role or "",
        *job.responsibilities,
        *job.achievements,
        *job.technologies,
    ])


def candidate_corpus(candidate: CandidateProfile) -> str:
    parts = [candidate.professional_summary]
    parts.extend(skill.name for skill in candidate.skills)

    for job in candidate.work_experience:
        parts.append(job_evidence_text(job))

    for project in candidate.projects:
        parts.extend([
            project.name,
            project.description,
            *project.technologies,
        ])

    for education in candidate.education:
        parts.extend([
            education.degree,
            education.institution,
            education.specialization or "",
        ])

    return " ".join(parts)


ML_CONCEPT_EVIDENCE = {
    "transformer architecture": [
        "transformer", "vision transformer", "attention mechanism",
    ],
    "training inference lifecycles": [
        "training lifecycle", "model training", "training pipeline",
        "inference lifecycle", "model inference", "inference pipeline",
    ],
    "optimization techniques": [
        "optimization technique", "model optimization", "performance tuning",
        "hyperparameter tuning", "quantization", "pruning", "gradient descent",
    ],
}


def concept_is_evidenced(corpus: str, concept: str) -> bool:
    """Match a named ML concept against explicit resume evidence/aliases."""
    aliases = ML_CONCEPT_EVIDENCE.get(normalise(concept), [concept])
    return any(contains_term(corpus, alias) for alias in aliases)


def matched_alternatives(
    requirement: Requirement,
    candidate: CandidateProfile,
    corpus: str,
) -> tuple[list[str], list[str]]:
    matched = []
    missing = []

    for alternative in requirement.alternatives:
        if requirement.category == "education":
            found = education_matches(candidate, alternative)
        elif requirement.category == "machine_learning" and normalise(alternative) in ML_CONCEPT_EVIDENCE:
            found = concept_is_evidenced(corpus, alternative)
        else:
            found = contains_term(corpus, alternative)

        if found:
            matched.append(alternative)
        else:
            missing.append(alternative)

    return matched, missing


def relevant_experience_years(
    requirement: Requirement,
    candidate: CandidateProfile,
) -> float:
    """Estimate relevant experience from dated roles and their descriptions.

    This is a heuristic estimate: a role is counted in full when its text
    contains at least one keyword associated with the requirement. It should
    not be interpreted as independently verified time spent on that activity.
    """
    name = requirement.name.lower()

    if "professional software development" in name:
        return total_experience_years(candidate)

    if "complex software systems" in name:
        keywords = [
            "software", "systems", "applications", "backend",
            "microservices", "development", "developed", "implemented",
        ]
    elif requirement.category == "architecture":
        keywords = [
            "architecture", "designed", "design", "scalability",
            "reliability", "system design", "technical design",
        ]
    elif requirement.category == "distributed_systems":
        keywords = [
            "distributed", "microservices", "multi-tier",
            "multi-threaded", "embedded", "large-scale",
            "services", "multitenancy",
        ]
    else:
        keywords = ["software development", "developed", "implemented", "engineered"]

    intervals = []

    for job in candidate.work_experience:
        evidence = normalise(job_evidence_text(job))
        if any(contains_term(evidence, keyword) for keyword in keywords):
            start = employment_month(job.start_date)
            end = employment_month(job.end_date, is_end=True)

            if start is not None and end is not None and end >= start:
                intervals.append((start, end))

    return round(calculate_months(intervals) / 12, 1)


def evaluate_requirement(
    requirement: Requirement,
    candidate: CandidateProfile,
    corpus: str,
) -> RequirementResult:
    matched_evidence = []
    missing_items = []
    component_scores = []

    # Evaluate alternatives such as Java OR C++.
    if requirement.alternatives:
        matched, missing = matched_alternatives(requirement, candidate, corpus)

        matched_evidence.extend(
            f"Matched alternative: {item}" for item in matched
        )

        if requirement.match_mode == "any":
            alternative_score = 100.0 if matched else 0.0
            # For OR conditions, unmatched alternatives are not gaps once
            # at least one acceptable alternative has satisfied the rule.
            if not matched:
                missing_items.extend(missing)
        else:
            alternative_score = (
                len(matched) / len(requirement.alternatives) * 100
                if requirement.alternatives else 0.0
            )
            missing_items.extend(missing)

        component_scores.append(alternative_score)

    # Evaluate explicit experience thresholds.
    if requirement.minimum_years is not None:
        years = relevant_experience_years(requirement, candidate)
        minimum_years = requirement.minimum_years

        if years >= minimum_years:
            experience_score = 100.0
            matched_evidence.append(
                f"Estimated {years} years of relevant experience "
                f"(minimum {minimum_years})"
            )
        else:
            experience_score = min(100.0, years / minimum_years * 100) if minimum_years > 0 else 0.0
            missing_items.append(
                f"{minimum_years} years required; {years} years estimated from resume"
            )

        component_scores.append(experience_score)

    # Fallback for requirements without alternatives or experience thresholds.
    if not component_scores:
        signals = requirement.evidence_signals or [requirement.name]
        matched_signals = [
            signal for signal in signals
            if contains_term(corpus, signal)
        ]
        score = (
            len(matched_signals) / len(signals) * 100
            if signals else 0.0
        )

        component_scores.append(score)
        matched_evidence.extend(matched_signals)

        missing_items.extend(
            signal for signal in signals if signal not in matched_signals
        )

    score = round(sum(component_scores) / len(component_scores), 1)

    if score == 100:
        status = "matched"
        explanation = "The resume contains evidence satisfying the evaluated conditions."
    elif score > 0:
        status = "partially_matched"
        explanation = "Some supporting evidence was found, but one or more conditions remain unmet."
    else:
        status = "not_evidenced"
        explanation = "The resume does not provide sufficient matching evidence for this requirement."

    return RequirementResult(
        requirement_name=requirement.name,
        category=requirement.category,
        priority=requirement.priority,
        status=status,
        score=score,
        matched_evidence=matched_evidence,
        missing_items=missing_items,
        explanation=explanation,
    )


def group_score(results: list[RequirementResult]) -> float:
    if not results:
        return 0.0

    return round(sum(result.score for result in results) / len(results), 1)


def match_candidate(
    candidate: CandidateProfile,
    job: JobProfile,
) -> MatchingResponse:
    corpus = candidate_corpus(candidate)

    mandatory_results = [
        evaluate_requirement(req, candidate, corpus)
        for req in job.mandatory_requirements
    ]
    preferred_results = [
        evaluate_requirement(req, candidate, corpus)
        for req in job.preferred_requirements
    ]

    mandatory_score = group_score(mandatory_results)
    preferred_score = group_score(preferred_results)

    if mandatory_results and preferred_results:
        overall_score = mandatory_score * 0.70 + preferred_score * 0.30
    elif mandatory_results:
        overall_score = mandatory_score
    elif preferred_results:
        overall_score = preferred_score
    else:
        overall_score = 0.0

    all_results = mandatory_results + preferred_results
    matched = [result for result in all_results if result.status == "matched"]
    partial = [result for result in all_results if result.status == "partially_matched"]
    missing = [result for result in all_results if result.status == "not_evidenced"]

    strengths = [result.requirement_name for result in matched]

    # Preserve order while removing duplicate gap labels.
    skill_gaps = list(dict.fromkeys(
        item
        for result in all_results
        for item in result.missing_items
    ))

    recommendations = [
        f"Add clear, verifiable resume evidence for: {result.requirement_name}"
        for result in partial + missing
    ]

    return MatchingResponse(
        match_score=round(overall_score, 1),
        mandatory_score=mandatory_score,
        preferred_score=preferred_score,
        total_experience_years=total_experience_years(candidate),
        matched_requirements=matched,
        partially_matched_requirements=partial,
        missing_requirements=missing,
        strengths=strengths,
        skill_gaps=skill_gaps,
        recommendations=recommendations,
        disclaimer=(
            "This score measures evidence found in the submitted resume. "
            "Experience durations are heuristic estimates based on resume dates "
            "and descriptions. Missing evidence does not prove the candidate "
            "lacks the skill or qualification."
        ),
    )
