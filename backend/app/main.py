
from fastapi import FastAPI

from app.api.analysis import router as analysis_router

from app.api.jobs import router as jobs_router

from app.api.resumes import router as resumes_router

from app.api.matching import router as matching_router


app = FastAPI(
    title="CareerLens AI",
    description="Local AI-powered Resume Matcher",
    version="1.0.0"
)

app.include_router(jobs_router)
app.include_router(resumes_router)
app.include_router(analysis_router)
app.include_router(matching_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "qwen3.5:4b"
    }
