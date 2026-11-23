import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.constants import MAX_RESUME_FILE_SIZE_BYTES
from app.main import app
from app.schemas.candidate import CandidateProfile
from app.api.resumes import extract_resume

client = TestClient(app)


def mock_candidate_profile(monkeypatch):
    async def extract_candidate_profile(resume_text):
        assert resume_text == "Extracted resume text " * 3
        return CandidateProfile(
            professional_summary="Test candidate profile",
        )

    monkeypatch.setattr(
        "app.api.resumes.extract_candidate_profile",
        extract_candidate_profile,
    )


@pytest.mark.parametrize(
    "file_size",
    [
        MAX_RESUME_FILE_SIZE_BYTES - 1,
        MAX_RESUME_FILE_SIZE_BYTES,
    ],
    ids=["one-byte-below-limit", "exact-limit"],
)
def test_resume_upload_accepts_file_at_or_below_limit(
    monkeypatch,
    file_size,
):
    monkeypatch.setattr(
        "app.api.resumes.extract_resume_text",
        lambda filename, content: "Extracted resume text " * 3,
    )
    mock_candidate_profile(monkeypatch)

    response = client.post(
        "/api/v1/resumes/extract",
        files={
            "resume": (
                "resume.pdf",
                b"x" * file_size,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["professional_summary"] == "Test candidate profile"


def test_resume_upload_rejects_file_one_byte_over_limit(monkeypatch):
    def unexpected_parser_call(*args, **kwargs):
        pytest.fail("Document parser must not run for an oversized upload.")

    monkeypatch.setattr(
        "app.api.resumes.extract_resume_text",
        unexpected_parser_call,
    )

    response = client.post(
        "/api/v1/resumes/extract",
        files={
            "resume": (
                "resume.pdf",
                b"x" * (MAX_RESUME_FILE_SIZE_BYTES + 1),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Resume exceeds the 5 MB limit."


def test_resume_upload_reads_only_limit_plus_one_byte():
    class TrackingUpload:
        filename = "resume.pdf"

        def __init__(self):
            self.read_size = None
            self.closed = False

        async def read(self, size=-1):
            self.read_size = size
            return b"x" * size

        async def close(self):
            self.closed = True

    upload = TrackingUpload()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(extract_resume(upload))

    assert exc_info.value.status_code == 413
    assert upload.read_size == MAX_RESUME_FILE_SIZE_BYTES + 1
    assert upload.closed is True


def test_resume_upload_rejects_missing_filename():
    response = client.post(
        "/api/v1/resumes/extract",
        files={
            "resume": (
                "",
                b"resume content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Resume filename is missing."


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("resume.txt", b"Some resume content " * 10),
        ("resume.pdf", b"This is not a PDF document."),
    ],
    ids=["unsupported-extension", "invalid-pdf-signature"],
)
def test_resume_upload_rejects_unsupported_or_invalid_documents(
    filename,
    content,
):
    response = client.post(
        "/api/v1/resumes/extract",
        files={
            "resume": (
                filename,
                content,
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 422


def test_analyze_upload_rejects_file_one_byte_over_limit(monkeypatch):
    def unexpected_parser_call(*args, **kwargs):
        pytest.fail("Document parser must not run for an oversized upload.")

    monkeypatch.setattr(
        "app.api.analysis.extract_resume_text",
        unexpected_parser_call,
    )

    response = client.post(
        "/api/v1/analyze-upload",
        data={
            "job_description": "A valid job description with more than fifty characters for testing."
        },
        files={
            "resume": (
                "resume.pdf",
                b"x" * (MAX_RESUME_FILE_SIZE_BYTES + 1),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Maximum resume size is 5 MB."
