"""
GradeOps – Pipeline Stage 4: HITL Review Dashboard API
FastAPI backend powering the high-throughput TA review dashboard.
TAs approve or override AI-proposed grades with keyboard shortcuts.
"""

from __future__ import annotations

import logging
import uuid
import shutil
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select, SQLModel

from schemas.models import (
    FinalGrade,
    GradingResult,
    TAReviewAction,
)
from database import engine, GradingResultRow, FinalGradeRow

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create tables on startup
    SQLModel.metadata.create_all(engine)
    yield

app = FastAPI(
    title="GradeOps HITL Review API",
    description="Human-in-the-Loop dashboard for approving AI-proposed grades.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for background pipeline job status
_job_status: dict[str, dict] = {}


# ─── Admin: Load Results Into Dashboard ──────────────────────────────────────

class BulkLoadRequest(BaseModel):
    results: list[GradingResult]


@app.post("/api/results/load", summary="Load grading results into the review queue")
def load_results(request: BulkLoadRequest) -> dict:
    """Called by the pipeline after Stage 3 completes."""
    with Session(engine) as session:
        for result in request.results:
            stmt = select(GradingResultRow).where(GradingResultRow.submission_id == result.submission_id)
            existing = session.exec(stmt).first()
            if existing:
                existing.student_id = result.student_id
                existing.rubric_id = result.rubric_id
                existing.exam_id = result.exam_id
                existing.total_score = result.total_score
                existing.max_score = result.max_score
                existing.criterion_results = [cr.model_dump() for cr in result.criterion_results]
                existing.plagiarism_flag = result.plagiarism_flag
                existing.plagiarism_similar_to = result.plagiarism_similar_to
                existing.evaluation_model = result.evaluation_model
                existing.status = result.status
                session.add(existing)
            else:
                row = GradingResultRow(
                    submission_id=result.submission_id,
                    student_id=result.student_id,
                    rubric_id=result.rubric_id,
                    exam_id=result.exam_id,
                    total_score=result.total_score,
                    max_score=result.max_score,
                    criterion_results=[cr.model_dump() for cr in result.criterion_results],
                    plagiarism_flag=result.plagiarism_flag,
                    plagiarism_similar_to=result.plagiarism_similar_to,
                    evaluation_model=result.evaluation_model,
                    status=result.status
                )
                session.add(row)
        session.commit()
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
    with Session(engine) as session:
        stmt = select(GradingResultRow).where(GradingResultRow.status == "proposed")
        if exam_id:
            stmt = stmt.where(GradingResultRow.exam_id == exam_id)
        if flagged_only:
            stmt = stmt.where(GradingResultRow.plagiarism_flag == True)
        
        results = session.exec(stmt).all()
        pending = list(results)
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
    with Session(engine) as session:
        stmt = select(GradingResultRow).where(GradingResultRow.submission_id == submission_id)
        result = session.exec(stmt).first()
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
    with Session(engine) as session:
        stmt = select(GradingResultRow).where(GradingResultRow.submission_id == action.submission_id)
        result = session.exec(stmt).first()
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
        
        # Save or update FinalGradeRow
        fg_stmt = select(FinalGradeRow).where(FinalGradeRow.submission_id == action.submission_id)
        existing_fg = session.exec(fg_stmt).first()
        if existing_fg:
            existing_fg.student_id = grade.student_id
            existing_fg.final_score = grade.final_score
            existing_fg.max_score = grade.max_score
            existing_fg.reviewed_by = grade.reviewed_by
            existing_fg.action_taken = grade.action_taken
            existing_fg.ai_proposed_score = grade.ai_proposed_score
            session.add(existing_fg)
        else:
            fg_row = FinalGradeRow(
                submission_id=grade.submission_id,
                student_id=grade.student_id,
                final_score=grade.final_score,
                max_score=grade.max_score,
                reviewed_by=grade.reviewed_by,
                action_taken=grade.action_taken,
                ai_proposed_score=grade.ai_proposed_score
            )
            session.add(fg_row)

        session.add(result)
        session.commit()

    logger.info(
        "TA %s %s submission %s: %.1f/%.1f",
        action.ta_id, action.action,
        action.submission_id, final_score, result.max_score,
    )
    return grade


@app.get(
    "/api/dashboard/stats",
    summary="Dashboard throughput and approval statistics",
)
def get_stats() -> dict:
    """High-level stats for the TA dashboard header."""
    with Session(engine) as session:
        results = session.exec(select(GradingResultRow)).all()
        total = len(results)
        approved = sum(1 for r in results if r.status == "approved")
        overridden = sum(1 for r in results if r.status == "overridden")
        pending = sum(1 for r in results if r.status == "proposed")
        flagged = sum(1 for r in results if r.plagiarism_flag)

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
    with Session(engine) as session:
        results = session.exec(select(FinalGradeRow)).all()
        return list(results)


# ─── Step 5: Missing API Routes ──────────────────────────────────────────────

@app.post("/api/upload", summary="Upload PDFs and rubric for an exam")
def upload_files(
    exam_id: str = Form(...),
    pdfs: list[UploadFile] = File(...),
    rubric: UploadFile = File(...),
) -> dict:
    """Accepts multipart files, saves them under storage/{exam_id}/, and returns info."""
    storage_dir = Path("storage") / exam_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Save rubric
    rubric_path = storage_dir / rubric.filename
    with open(rubric_path, "wb") as f:
        shutil.copyfileobj(rubric.file, f)
        
    # Save PDFs
    for pdf in pdfs:
        pdf_path = storage_dir / pdf.filename
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)
            
    return {
        "exam_id": exam_id,
        "pdf_count": len(pdfs),
        "rubric_path": str(rubric_path),
    }


class PipelineRunRequest(BaseModel):
    exam_id: str
    rubric_path: str
    vlm_backend: str = "auto"


def background_pipeline_task(job_id: str, request: PipelineRunRequest):
    from orchestrator import run_pipeline, PipelineConfig
    
    _job_status[job_id]["stage"] = "Starting Pipeline"
    _job_status[job_id]["progress_pct"] = 10
    
    try:
        pdf_dir = Path("storage") / request.exam_id
        
        config = PipelineConfig(
            rubric_json_path=request.rubric_path,
            exam_pdf_dir=pdf_dir,
            exam_id=request.exam_id,
            vlm_backend=request.vlm_backend,
        )
        
        _job_status[job_id]["stage"] = "Running Pipeline"
        _job_status[job_id]["progress_pct"] = 40
        
        run_pipeline(config)
        
        _job_status[job_id]["status"] = "complete"
        _job_status[job_id]["stage"] = "Finished"
        _job_status[job_id]["progress_pct"] = 100
    except Exception as e:
        logger.error("Pipeline job %s failed: %s", job_id, e)
        _job_status[job_id]["status"] = "failed"
        _job_status[job_id]["stage"] = f"Error: {str(e)}"
        _job_status[job_id]["progress_pct"] = 100


@app.post("/api/pipeline/run", summary="Start pipeline execution in the background")
def run_pipeline_job(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    job_id = str(uuid.uuid4())
    _job_status[job_id] = {
        "job_id": job_id,
        "status": "running",
        "stage": "Initialized",
        "progress_pct": 0,
    }
    background_tasks.add_task(background_pipeline_task, job_id, request)
    return {"job_id": job_id}


@app.get("/api/pipeline/status/{job_id}", summary="Get background job status")
def get_pipeline_status(job_id: str) -> dict:
    status = _job_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


class PlagiarismAlertResponse(BaseModel):
    submission_id: str
    student_id: str
    plagiarism_similar_to: list[str]


@app.get(
    "/api/plagiarism/alerts",
    response_model=list[PlagiarismAlertResponse],
    summary="Get all plagiarism flagged submissions",
)
def get_plagiarism_alerts() -> list:
    with Session(engine) as session:
        stmt = select(GradingResultRow).where(GradingResultRow.plagiarism_flag == True)
        results = session.exec(stmt).all()
        return [
            {
                "submission_id": r.submission_id,
                "student_id": r.student_id,
                "plagiarism_similar_to": r.plagiarism_similar_to
            }
            for r in results
        ]
