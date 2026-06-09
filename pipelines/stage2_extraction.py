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

    Install: pip install nougat-ocr transformers torch
    """
    try:
        from transformers import NougatProcessor, VisionEncoderDecoderModel  # type: ignore
        import torch
    except ImportError as e:
        raise ImportError(
            "Install Nougat dependencies: pip install nougat-ocr transformers torch accelerate"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load processor and model from HuggingFace
    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
    model = model.to(device)
    model.eval()

    results = []
    for img in images:
        try:
            # Process image and generate text
            pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    max_length=3000,
                    early_stopping=True,
                )
            # Decode the generated tokens
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
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
) -> ExtractedSubmission:
    """
    Run the full VLM extraction pipeline on a student submission.

    Args:
        submission: Ingested StudentSubmission with status='ingested'.
        backend:    Which VLM to use. 'auto' selects based on content.

    Returns:
        ExtractedSubmission with per-page transcribed answers.
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

    answers = [
        ExtractedAnswer(
            criterion_name=f"page_{i+1}",  # refined downstream by agentic stage
            raw_text=text.strip(),
            confidence=0.90,               # production: get logit confidence
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
    logger.info("Extraction complete: %d answers", len(answers))
    return extracted


# ─── Batch Extraction ─────────────────────────────────────────────────────────

def batch_extract(
    submissions: list[StudentSubmission],
    backend: VLMBackend = "auto",
    max_workers: int = 4,
) -> list[ExtractedSubmission]:
    """
    Extract a batch of submissions in parallel using a thread pool.

    Args:
        submissions:  List of ingested submissions.
        backend:      VLM backend to use.
        max_workers:  Thread pool size (GPU-bound: keep low).

    Returns:
        List of ExtractedSubmission objects in the same order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, ExtractedSubmission] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(extract_submission, sub, backend): sub.submission_id
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
