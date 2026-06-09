"""
GradeOps – Pipeline Stage 2: VLM Translation Engine
Extracts and transcribes handwritten answers from exam PDFs using
Nougat (document/LaTeX-heavy) and Qwen-VL (general handwriting).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF – pip install pymupdf
from PIL import Image
import io

from schemas.models import ExtractedAnswer, ExtractedSubmission, StudentSubmission

logger = logging.getLogger(__name__)

VLMBackend = Literal["nougat", "qwen-vl", "auto"]


# ─── Criterion Matching ────────────────────────────────────────────────────────

def segment_answers_by_criterion(
    page_texts: list[str],
    criteria: list,  # list[RubricCriterion]
) -> dict[str, str]:
    """
    Use an LLM to intelligently map extracted page texts to rubric criteria.
    Handles students who skip questions or answer out of order.

    Args:
        page_texts: List of extracted text from each PDF page.
        criteria:   List of RubricCriterion objects from the rubric.

    Returns:
        Dict mapping criterion.name -> answer_text for that criterion.
        If a criterion has no answer, value is empty string.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as e:
        raise ImportError(
            "Install langchain dependencies: pip install langchain-openai langchain"
        ) from e

    if not criteria or not page_texts:
        return {c.name: "" for c in criteria}

    # Build the LLM prompt
    criterion_list = "\n".join(
        f"- {c.name}: {c.description}" for c in criteria
    )
    page_list = "\n".join(
        f"Page {i+1}:\n{text[:500]}..." if len(text) > 500 else f"Page {i+1}:\n{text}"
        for i, text in enumerate(page_texts)
    )

    system_prompt = """You are an expert academic test grader. Your task is to read
extracted page texts from a student's exam and match each page (or group of pages)
to the corresponding rubric criterion.

Return ONLY a valid JSON object with this exact schema:
{
  "criterion_name_1": "extracted text for that criterion",
  "criterion_name_2": "extracted text for that criterion",
  ...
}

Rules:
1. Each criterion name must exactly match one from the provided list.
2. If a criterion is not answered, use empty string as its value.
3. Combine multiple pages if they belong to the same criterion.
4. Return ONLY valid JSON—no preamble or explanation.
"""

    user_prompt = f"""RUBRIC CRITERIA:
{criterion_list}

EXTRACTED PAGE TEXTS:
{page_list}

Match each page text to the appropriate criterion and return the mapping as JSON."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    
    try:
        import json
        answer_map = json.loads(response.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse criterion mapping LLM response: %s", e)
        # Fallback: map by position (the old buggy behavior)
        answer_map = {}
        for i, criterion in enumerate(criteria):
            answer_map[criterion.name] = page_texts[i] if i < len(page_texts) else ""

    # Ensure all criteria are in the map with at least empty string
    for criterion in criteria:
        if criterion.name not in answer_map:
            answer_map[criterion.name] = ""

    return answer_map


# ─── PDF → Images ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """
    Rasterise every page of a PDF into a PIL Image.

    Args:
        pdf_path: Path to the scanned exam PDF.
        dpi:      Render resolution (200 dpi is sufficient for OCR).

    Returns:
        Ordered list of PIL Images, one per page.
    """
    doc = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    zoom = dpi / 72  # fitz native resolution is 72 dpi
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


# ─── Nougat Backend ───────────────────────────────────────────────────────────

def _extract_with_nougat(images: list[Image.Image]) -> list[str]:
    """
    Use facebook/nougat-base for LaTeX-rich academic documents.
    Best for typed or printed math-heavy exams.

    Install: pip install nougat-ocr
    """
    try:
        from nougat import NougatModel  # type: ignore
    except ImportError as e:
        raise ImportError("Install nougat-ocr: pip install nougat-ocr") from e

    model = NougatModel.from_pretrained("facebook/nougat-base")

    results = []
    for img in images:
        try:
            # predict() returns a string with the extracted text
            text = model.predict(img)
            results.append(text)
        except Exception as e:
            logger.warning("Nougat extraction failed for image: %s", e)
            results.append("")
    
    return results


# ─── Qwen-VL Backend ─────────────────────────────────────────────────────────

def _extract_with_qwen_vl(images: list[Image.Image]) -> list[str]:
    """
    Use Qwen/Qwen-VL-Chat for general handwritten text and mixed content.
    Best for messy handwritten answers.

    Install: pip install transformers accelerate
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch
    except ImportError as e:
        raise ImportError("Install transformers: pip install transformers accelerate") from e

    model_id = "Qwen/Qwen-VL-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto"
    ).eval()

    results = []
    for img in images:
        query = tokenizer.from_list_format([
            {"image": img},
            {"text": (
                "Transcribe ALL handwritten text and mathematical expressions "
                "exactly as written. Use LaTeX for equations. "
                "Preserve step numbering and structure."
            )},
        ])
        response, _ = model.chat(tokenizer, query=query, history=None)
        results.append(response)
    return results


# ─── Auto-Select Backend ─────────────────────────────────────────────────────

def _detect_content_type(images: list[Image.Image]) -> VLMBackend:
    """
    Heuristic: if the page appears to have dense print/LaTeX use Nougat,
    otherwise use Qwen-VL for handwriting.
    (Production: train a lightweight classifier.)
    """
    # Simple heuristic – prefer Qwen-VL as the safe default for handwriting
    return "qwen-vl"


# ─── Main Extraction Entry Point ─────────────────────────────────────────────

def extract_submission(
    submission: StudentSubmission,
    backend: VLMBackend = "auto",
    rubric_criteria: list | None = None,
) -> ExtractedSubmission:
    """
    Run the full VLM extraction pipeline on a student submission.

    Args:
        submission:      Ingested StudentSubmission with status='ingested'.
        backend:         Which VLM to use. 'auto' selects based on content.
        rubric_criteria: Optional list of RubricCriterion for intelligent answer matching.
                         If provided, uses LLM to match answers to criteria.

    Returns:
        ExtractedSubmission with per-criterion mapped answers.
    """
    pdf_path = Path(submission.pdf_path)
    images = pdf_to_images(pdf_path)

    if backend == "auto":
        backend = _detect_content_type(images)

    logger.info(
        "Extracting submission=%s with backend=%s (%d pages)",
        submission.submission_id, backend, len(images),
    )

    if backend == "nougat":
        raw_texts = _extract_with_nougat(images)
        model_name = "facebook/nougat-base"
    else:
        raw_texts = _extract_with_qwen_vl(images)
        model_name = "Qwen/Qwen-VL-Chat"

    # Map answers to criteria intelligently if rubric is provided
    if rubric_criteria:
        criterion_map = segment_answers_by_criterion(raw_texts, rubric_criteria)
        answers = [
            ExtractedAnswer(
                criterion_name=criterion.name,
                raw_text=criterion_map.get(criterion.name, "").strip(),
                confidence=0.90,
                page_number=0,  # 0 indicates criterion-based, not page-based
            )
            for criterion in rubric_criteria
        ]
    else:
        # Fallback to page-based answers if no rubric provided
        answers = [
            ExtractedAnswer(
                criterion_name=f"page_{i+1}",
                raw_text=text.strip(),
                confidence=0.90,
                page_number=i + 1,
            )
            for i, text in enumerate(raw_texts)
        ]

    extracted = ExtractedSubmission(
        submission_id=submission.submission_id,
        student_id=submission.student_id,
        rubric_id=submission.rubric_id,
        answers=answers,
        extraction_model=model_name,
    )
    logger.info("Extraction complete: %d answers mapped to criteria", len(answers))
    return extracted


# ─── Batch Extraction ─────────────────────────────────────────────────────────

def batch_extract(
    submissions: list[StudentSubmission],
    backend: VLMBackend = "auto",
    max_workers: int = 4,
    rubric_criteria: list | None = None,
) -> list[ExtractedSubmission]:
    """
    Extract a batch of submissions in parallel using a thread pool.

    Args:
        submissions:      List of ingested submissions.
        backend:          VLM backend to use.
        max_workers:      Thread pool size (GPU-bound: keep low).
        rubric_criteria:  Optional list of RubricCriterion for intelligent matching.

    Returns:
        List of ExtractedSubmission objects in the same order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, ExtractedSubmission] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(extract_submission, sub, backend, rubric_criteria): sub.submission_id
            for sub in submissions
        }
        for future in as_completed(futures):
            sid = futures[future]
            try:
                results[sid] = future.result()
                logger.info("Extracted: %s ✓", sid)
            except Exception as exc:
                logger.error("Extraction failed for %s: %s", sid, exc)

    # Return in original order
    return [results[sub.submission_id] for sub in submissions if sub.submission_id in results]
