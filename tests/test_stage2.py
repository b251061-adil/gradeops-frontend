import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import fitz
from PIL import Image

from pipelines.stage2_extraction import (
    pdf_to_images,
    extract_submission,
    batch_extract,
    segment_answers_by_criterion,
)
from schemas.models import StudentSubmission, RubricCriterion


@pytest.fixture
def dummy_pdf_file(tmp_path):
    pdf_path = tmp_path / "test_two_pages.pdf"
    doc = fitz.open()
    doc.new_page()  # Page 1
    doc.new_page()  # Page 2
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def dummy_submission(dummy_pdf_file):
    return StudentSubmission(
        submission_id="sub_123",
        student_id="student_123",
        exam_id="EXAM-1",
        rubric_id="RUBRIC-1",
        pdf_path=str(dummy_pdf_file),
        status="ingested",
    )


def test_pdf_to_images(dummy_pdf_file):
    images = pdf_to_images(dummy_pdf_file)
    assert len(images) == 2
    for img in images:
        assert isinstance(img, Image.Image)


def test_extract_submission_mock(dummy_submission):
    # Without rubric criteria, maps page_number
    extracted = extract_submission(dummy_submission, backend="mock")
    assert extracted.submission_id == "sub_123"
    assert len(extracted.answers) == 2
    assert extracted.answers[0].criterion_name == "page_1"
    assert "Problem 1" in extracted.answers[0].raw_text
    assert extracted.answers[1].criterion_name == "page_2"
    assert "Problem 2" in extracted.answers[1].raw_text

    # With rubric criteria
    criteria = [
        RubricCriterion(name="Problem 1", max_points=10, description="desc1"),
        RubricCriterion(name="Problem 2", max_points=5, description="desc2"),
    ]
    with patch("pipelines.stage2_extraction.segment_answers_by_criterion") as mock_segment:
        mock_segment.return_value = {
            "Problem 1": "Mocked P1 answer",
            "Problem 2": "Mocked P2 answer",
        }
        extracted_with_criteria = extract_submission(
            dummy_submission, backend="mock", rubric_criteria=criteria
        )
        assert len(extracted_with_criteria.answers) == 2
        assert extracted_with_criteria.answers[0].criterion_name == "Problem 1"
        assert extracted_with_criteria.answers[0].raw_text == "Mocked P1 answer"


def test_batch_extract(dummy_submission):
    submissions = [dummy_submission]
    extracted_list = batch_extract(submissions, backend="mock", max_workers=2)
    assert len(extracted_list) == 1
    assert extracted_list[0].submission_id == "sub_123"
    # Status should be transitioned to "extracted"
    assert dummy_submission.status == "extracted"


@patch("langchain_openai.ChatOpenAI")
def test_extract_with_openai_mocked(mock_chat_class, dummy_submission):
    mock_llm = MagicMock()
    mock_chat_class.return_value = mock_llm
    
    mock_response = MagicMock()
    mock_response.content = "OpenAI Transcribed Page Text"
    mock_llm.invoke.return_value = mock_response

    extracted = extract_submission(dummy_submission, backend="openai")
    assert extracted.extraction_model == "gpt-4o"
    assert len(extracted.answers) == 2
    assert extracted.answers[0].raw_text == "OpenAI Transcribed Page Text"
