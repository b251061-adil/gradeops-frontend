"""
GradeOps – Core Pydantic Schemas
Covers: JSON rubrics, student submissions, extracted text, and grading results.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─── Rubric Schemas ──────────────────────────────────────────────────────────

class RubricCondition(BaseModel):
    """A single scorable condition within a rubric criterion."""
    condition_id: str
    description: str
    points: float
    is_partial_credit: bool = False


class RubricCriterion(BaseModel):
    """One graded criterion (e.g., 'Problem 1 – correct approach')."""
    name: str
    max_points: float
    description: str
    conditions: list[RubricCondition] = Field(default_factory=list)


class GradingRubric(BaseModel):
    """Complete rubric uploaded by an instructor."""
    rubric_id: str
    course_id: str
    exam_name: str
    total_points: float
    criteria: list[RubricCriterion]


# ─── Submission Schemas ───────────────────────────────────────────────────────

class StudentSubmission(BaseModel):
    """Raw submission record created after bulk ingestion."""
    submission_id: str
    student_id: str
    exam_id: str
    rubric_id: str
    pdf_path: str                   # path to the scanned PDF
    status: Literal[
        "ingested", "extracted", "evaluated", "reviewed", "finalized"
    ] = "ingested"


class ExtractedAnswer(BaseModel):
    """Transcribed answer for a single criterion after VLM extraction."""
    criterion_name: str
    raw_text: str                   # OCR output (may include LaTeX)
    confidence: float = Field(ge=0.0, le=1.0)
    page_number: int


class ExtractedSubmission(BaseModel):
    """All extracted answers for one student submission."""
    submission_id: str
    student_id: str
    rubric_id: str
    answers: list[ExtractedAnswer]
    extraction_model: str           # e.g. "Nougat" or "Qwen-VL"


# ─── Grading Result Schemas ───────────────────────────────────────────────────

class ConditionResult(BaseModel):
    """Outcome of evaluating a single rubric condition."""
    condition_id: str
    met: bool
    partial: bool = False
    points_awarded: float
    reasoning: str


class CriterionResult(BaseModel):
    """Aggregated result for one rubric criterion."""
    criterion_name: str
    points_awarded: float
    max_points: float
    condition_results: list[ConditionResult]
    justification: str              # structured textual rationale


class GradingResult(BaseModel):
    """Complete AI-proposed grading output for one submission."""
    submission_id: str
    student_id: str
    rubric_id: str
    exam_id: Optional[str] = None
    total_score: float
    max_score: float
    criterion_results: list[CriterionResult]
    plagiarism_flag: bool = False
    plagiarism_similar_to: list[str] = Field(default_factory=list)
    evaluation_model: str
    status: Literal["proposed", "approved", "overridden"] = "proposed"


# ─── HITL Review Schemas ──────────────────────────────────────────────────────

class TAReviewAction(BaseModel):
    """Action submitted by a TA from the review dashboard."""
    submission_id: str
    ta_id: str
    action: Literal["approve", "override"]
    override_score: Optional[float] = None
    override_notes: Optional[str] = None


class FinalGrade(BaseModel):
    """Finalized grade after TA review."""
    submission_id: str
    student_id: str
    final_score: float
    max_score: float
    reviewed_by: str
    action_taken: Literal["approve", "override"]
    ai_proposed_score: float
