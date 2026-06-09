"""
GradeOps – Master Orchestrator
Chains Stage 1 → 2 → 3a → 3b → 4 into a single end-to-end pipeline run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pipelines.stage1_ingestion import bulk_ingest_directory, ingest_rubric
from pipelines.stage2_extraction import batch_extract
from pipelines.stage3a_evaluation import batch_evaluate
from pipelines.stage3b_plagiarism import apply_plagiarism_flags, detect_plagiarism
from schemas.models import FinalGrade, GradingResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a full GradeOps pipeline run."""
    rubric_json_path: str | Path         # instructor-provided JSON rubric
    exam_pdf_dir: str | Path             # directory of scanned PDFs
    exam_id: str                         # e.g. "MTH101-MIDTERM-2025"
    vlm_backend: str = "auto"            # "nougat" | "qwen-vl" | "openai" | "mock" | "auto"
    plagiarism_threshold: float = 0.82
    max_extract_workers: int = 4


@dataclass
class PipelineOutput:
    """Outputs produced by a full pipeline run."""
    grading_results: list[GradingResult]
    plagiarism_alerts: list
    stats: dict


def run_pipeline(config: PipelineConfig) -> PipelineOutput:
    """
    Execute the complete GradeOps Agentic Grading Pipeline.

    Stage 1 – Bulk Ingestion:   Load rubric + register all exam PDFs.
    Stage 2 – VLM Extraction:   Transcribe handwritten answers (Nougat/Qwen-VL).
    Stage 3a – LLM Evaluation:  Award partial credit via LangGraph.
    Stage 3b – Plagiarism:      Detect structural collusion across submissions.
    Stage 4  – HITL Queue:      Push results to TA review dashboard API.

    Args:
        config: PipelineConfig specifying all runtime parameters.

    Returns:
        PipelineOutput containing grading results, alerts, and stats.
    """

    # ── Stage 1: Ingestion ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 – Bulk Ingestion")
    logger.info("=" * 60)

    rubric = ingest_rubric(config.rubric_json_path)
    logger.info("Rubric loaded: %s (%d criteria)", rubric.rubric_id, len(rubric.criteria))

    submissions = bulk_ingest_directory(
        pdf_dir=config.exam_pdf_dir,
        exam_id=config.exam_id,
        rubric_id=rubric.rubric_id,
    )
    logger.info("Submissions ingested: %d", len(submissions))

    # ── Stage 2: VLM Extraction ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 – VLM Translation Engine (backend=%s)", config.vlm_backend)
    logger.info("=" * 60)

    extracted_list = batch_extract(
        submissions=submissions,
        backend=config.vlm_backend,
        max_workers=config.max_extract_workers,
        rubric_criteria=rubric.criteria,  # Pass rubric criteria for intelligent matching
    )
    logger.info("Extracted: %d/%d submissions", len(extracted_list), len(submissions))

    # ── Stage 3a: Agentic Evaluation ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3a – Agentic LLM Cognitive Engine")
    logger.info("=" * 60)

    grading_results = batch_evaluate(extracted_list, rubric, submissions=submissions)
    logger.info("Evaluated: %d submissions", len(grading_results))

    avg_score = (
        sum(r.total_score for r in grading_results) /
        max(len(grading_results), 1)
    )
    logger.info("Average score: %.2f / %.2f", avg_score, rubric.total_points)

    # ── Stage 3b: Plagiarism Detection ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3b – Structural Plagiarism Detection")
    logger.info("=" * 60)

    alerts = detect_plagiarism(extracted_list, threshold=config.plagiarism_threshold)
    grading_results = apply_plagiarism_flags(grading_results, alerts)

    flagged_count = sum(1 for r in grading_results if r.plagiarism_flag)
    logger.info(
        "Plagiarism: %d alerts | %d submissions flagged",
        len(alerts), flagged_count,
    )

    # ── Stage 4: Push to HITL Dashboard ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 – Loading results into HITL Review Dashboard")
    logger.info("=" * 60)

    import httpx
    try:
        resp = httpx.post(
            "http://localhost:8000/api/results/load",
            json={"results": [r.model_dump() for r in grading_results]},
            timeout=30,
        )
        resp.raise_for_status()
        logger.info("Dashboard loaded: %s", resp.json())
    except Exception as exc:
        logger.warning("Could not push to dashboard (is the API running?): %s", exc)

    stats = {
        "exam_id": config.exam_id,
        "total_submissions": len(submissions),
        "extracted": len(extracted_list),
        "evaluated": len(grading_results),
        "plagiarism_alerts": len(alerts),
        "flagged_submissions": flagged_count,
        "average_score": round(avg_score, 2),
        "max_score": rubric.total_points,
    }

    logger.info("Pipeline complete. Stats: %s", stats)
    return PipelineOutput(
        grading_results=grading_results,
        plagiarism_alerts=alerts,
        stats=stats,
    )


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the GradeOps pipeline")
    parser.add_argument("--rubric", required=True, help="Path to JSON rubric file")
    parser.add_argument("--pdfs", required=True, help="Directory of exam PDF scans")
    parser.add_argument("--exam-id", required=True, help="Exam identifier")
    parser.add_argument("--backend", default="auto", choices=["nougat", "qwen-vl", "openai", "mock", "auto"])
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()

    config = PipelineConfig(
        rubric_json_path=args.rubric,
        exam_pdf_dir=args.pdfs,
        exam_id=args.exam_id,
        vlm_backend=args.backend,
        plagiarism_threshold=args.threshold,
    )
    output = run_pipeline(config)
    print("\n── Pipeline Stats ──")
    for k, v in output.stats.items():
        print(f"  {k}: {v}")
