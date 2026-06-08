"""
GradeOps – Pipeline Stage 1: Bulk Ingestion
Handles uploading of exam scan PDFs and instructor-defined JSON rubrics.
"""

import json
import logging
import shutil
import uuid
from pathlib import Path

from schemas.models import GradingRubric, StudentSubmission

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("storage/uploads")
RUBRIC_DIR = Path("storage/rubrics")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUBRIC_DIR.mkdir(parents=True, exist_ok=True)


# ─── Rubric Ingestion ─────────────────────────────────────────────────────────

def ingest_rubric(rubric_json_path: str | Path) -> GradingRubric:
    """
    Load and validate an instructor-provided JSON rubric file.

    Args:
        rubric_json_path: Path to the strict JSON rubric file.

    Returns:
        Validated GradingRubric instance.

    Raises:
        ValueError: If the rubric JSON is malformed or missing required fields.
    """
    path = Path(rubric_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    rubric = GradingRubric(**raw)

    # Persist a copy to the rubric store
    dest = RUBRIC_DIR / f"{rubric.rubric_id}.json"
    shutil.copy(path, dest)
    logger.info("Rubric ingested: %s → %s", rubric.rubric_id, dest)
    return rubric


def load_rubric(rubric_id: str) -> GradingRubric:
    """Load a previously ingested rubric by ID."""
    path = RUBRIC_DIR / f"{rubric_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found in store: {rubric_id}")
    with open(path) as f:
        return GradingRubric(**json.load(f))


# ─── PDF Ingestion ────────────────────────────────────────────────────────────

def ingest_exam_pdf(
    pdf_path: str | Path,
    student_id: str,
    exam_id: str,
    rubric_id: str,
) -> StudentSubmission:
    """
    Register a single exam scan PDF as a StudentSubmission.

    Args:
        pdf_path:   Path to the scanned PDF.
        student_id: Anonymised student identifier.
        exam_id:    Exam/course batch identifier.
        rubric_id:  ID of the rubric to use for grading.

    Returns:
        A StudentSubmission with status='ingested'.
    """
    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(f"PDF not found: {src}")

    submission_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{submission_id}.pdf"
    shutil.copy(src, dest)

    submission = StudentSubmission(
        submission_id=submission_id,
        student_id=student_id,
        exam_id=exam_id,
        rubric_id=rubric_id,
        pdf_path=str(dest),
        status="ingested",
    )
    logger.info("PDF ingested: student=%s → submission=%s", student_id, submission_id)
    return submission


def bulk_ingest_directory(
    pdf_dir: str | Path,
    exam_id: str,
    rubric_id: str,
    student_id_fn=None,
) -> list[StudentSubmission]:
    """
    Ingest all PDFs in a directory as a batch.

    Args:
        pdf_dir:       Directory containing scanned PDF exams.
        exam_id:       Shared exam identifier for the batch.
        rubric_id:     Rubric to apply to every submission.
        student_id_fn: Optional callable(Path) → str to derive student IDs
                       from filenames. Defaults to using the stem.

    Returns:
        List of StudentSubmission objects, one per PDF.
    """
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found in: {pdf_dir}")

    if student_id_fn is None:
        student_id_fn = lambda p: p.stem  # noqa: E731

    submissions: list[StudentSubmission] = []
    for pdf in pdfs:
        student_id = student_id_fn(pdf)
        sub = ingest_exam_pdf(pdf, student_id, exam_id, rubric_id)
        submissions.append(sub)

    logger.info(
        "Bulk ingestion complete: %d submissions for exam=%s", len(submissions), exam_id
    )
    return submissions
