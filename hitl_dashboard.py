"""
GradeOps – Pipeline Stage 4: HITL Review Dashboard API
FastAPI backend powering the high-throughput TA review dashboard.
TAs approve or override AI-proposed grades with keyboard shortcuts.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas.models import (
    FinalGrade,
    GradingResult,
    TAReviewAction,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GradeOps HITL Review API",
    description="Human-in-the-Loop dashboard for approving AI-proposed grades.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-Memory Store (replace with PostgreSQL / MongoDB in production) ─────────

_grading_results: dict[str, GradingResult] = {}
_final_grades: dict[str, FinalGrade] = {}


# ─── Admin: Load Results Into Dashboard ──────────────────────────────────────

class BulkLoadRequest(BaseModel):
    results: list[GradingResult]


@app.post("/api/results/load", summary="Load grading results into the review queue")
def load_results(request: BulkLoadRequest) -> dict:
    """Called by the pipeline after Stage 3 completes."""
    for result in request.results:
        _grading_results[result.submission_id] = result
    return {"loaded": len(request.results)}


# ─── TA Dashboard Endpoints ───────────────────────────────────────────────────

@app.get(
    "/api/dashboard/queue",
    response_model=list[GradingResult],
    summary="Get all submissions pending TA review",
)
def get_review_queue(
    exam_id: Optional[str] = Query(None),
    flagged_only: bool = Query(False, description="Show only plagiarism-flagged submissions"),
) -> list[GradingResult]:
    """Returns submissions in 'proposed' status, ordered for rapid review."""
    pending = [
        r for r in _grading_results.values()
        if r.status == "proposed"
    ]
    if exam_id:
        pending = [r for r in pending if r.exam_id == exam_id]
    if flagged_only:
        pending = [r for r in pending if r.plagiarism_flag]
    # Sort: flagged first, then by submission ID for determinism
    pending.sort(key=lambda r: (not r.plagiarism_flag, r.submission_id))
    return pending


@app.get(
    "/api/dashboard/submission/{submission_id}",
    response_model=GradingResult,
    summary="Get a single submission for side-by-side review",
)
def get_submission(submission_id: str) -> GradingResult:
    """Returns AI-proposed grade and justification for a single submission."""
    result = _grading_results.get(submission_id)
    if not result:
        raise HTTPException(status_code=404, detail="Submission not found")
    return result


@app.post(
    "/api/dashboard/review",
    response_model=FinalGrade,
    summary="TA approves or overrides an AI-proposed grade",
)
def submit_review(action: TAReviewAction) -> FinalGrade:
    """
    Process a TA review action (approve or override).
    Triggered by [ENTER] (approve) or [SPACE] (override) in the dashboard.
    """
    result = _grading_results.get(action.submission_id)
    if not result:
        raise HTTPException(status_code=404, detail="Submission not found")

    if result.status != "proposed":
        raise HTTPException(
            status_code=409,
            detail=f"Submission already reviewed (status={result.status})",
        )

    if action.action == "approve":
        final_score = result.total_score
        result.status = "approved"
    else:  # override
        if action.override_score is None:
            raise HTTPException(
                status_code=422,
                detail="override_score is required when action='override'",
            )
        final_score = action.override_score
        result.status = "overridden"

    grade = FinalGrade(
        submission_id=result.submission_id,
        student_id=result.student_id,
        final_score=final_score,
        max_score=result.max_score,
        reviewed_by=action.ta_id,
        action_taken=action.action,
        ai_proposed_score=result.total_score,
    )
    _final_grades[result.submission_id] = grade

    logger.info(
        "TA %s %s submission %s: %.1f/%.1f",
        action.ta_id, action.action,
        result.submission_id, final_score, result.max_score,
    )
    return grade


@app.get(
    "/api/dashboard/stats",
    summary="Dashboard throughput and approval statistics",
)
def get_stats() -> dict:
    """High-level stats for the TA dashboard header."""
    total = len(_grading_results)
    approved = sum(1 for r in _grading_results.values() if r.status == "approved")
    overridden = sum(1 for r in _grading_results.values() if r.status == "overridden")
    pending = sum(1 for r in _grading_results.values() if r.status == "proposed")
    flagged = sum(1 for r in _grading_results.values() if r.plagiarism_flag)

    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "override_rate_pct": round(overridden / max(approved + overridden, 1) * 100, 1),
        "plagiarism_flagged": flagged,
    }


@app.get(
    "/api/results/final",
    response_model=list[FinalGrade],
    summary="Export all finalized grades",
)
def get_final_grades() -> list[FinalGrade]:
    """Returns all finalized grades for export to the LMS."""
    return list(_final_grades.values())
