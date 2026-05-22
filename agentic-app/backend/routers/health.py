"""
health.py — Health check endpoint.
Used as a liveness probe (Docker, load balancers, CI).
"""
from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter(prefix="/health", tags=["health"])

_start_time = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str = "1.0.0"


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def health_check() -> HealthResponse:
    """Returns OK if the server is alive."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 2),
    )
