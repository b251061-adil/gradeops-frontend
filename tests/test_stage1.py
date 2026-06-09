import json
import pytest
from pathlib import Path
import fitz

from pipelines.stage1_ingestion import (
    ingest_rubric,
    load_rubric,
    ingest_exam_pdf,
    bulk_ingest_directory,
    UPLOAD_DIR,
    RUBRIC_DIR,
)
from schemas.models import GradingRubric

@pytest.fixture
def temp_dirs(tmp_path, monkeypatch):
    # Redirect UPLOAD_DIR and RUBRIC_DIR to a temp directory
    temp_upload = tmp_path / "uploads"
    temp_rubrics = tmp_path / "rubrics"
    temp_upload.mkdir()
    temp_rubrics.mkdir()

    monkeypatch.setattr("pipelines.stage1_ingestion.UPLOAD_DIR", temp_upload)
    monkeypatch.setattr("pipelines.stage1_ingestion.RUBRIC_DIR", temp_rubrics)
    return temp_upload, temp_rubrics


@pytest.fixture
def dummy_rubric_file(tmp_path):
    rubric_data = {
        "rubric_id": "TEST-RUBRIC-1",
        "course_id": "TEST101",
        "exam_name": "Test Exam",
        "total_points": 10.0,
        "criteria": [
            {
                "name": "Problem 1",
                "max_points": 10.0,
                "description": "Solve problem 1",
                "conditions": [
                    {
                        "condition_id": "P1_C1",
                        "description": "Correct formulation",
                        "points": 5.0,
                        "is_partial_credit": False
                    },
                    {
                        "condition_id": "P1_C2",
                        "description": "Correct calculation",
                        "points": 5.0,
                        "is_partial_credit": True
                    }
                ]
            }
        ]
    }
    file_path = tmp_path / "test_rubric.json"
    with open(file_path, "w") as f:
        json.dump(rubric_data, f)
    return file_path


@pytest.fixture
def dummy_pdf_file(tmp_path):
    pdf_path = tmp_path / "student_123.pdf"
    doc = fitz.open()
    doc.new_page()  # Add a blank page
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_ingest_rubric(dummy_rubric_file, temp_dirs):
    _, temp_rubrics = temp_dirs
    rubric = ingest_rubric(dummy_rubric_file)
    assert isinstance(rubric, GradingRubric)
    assert rubric.rubric_id == "TEST-RUBRIC-1"
    
    # Check if a copy was saved to the store
    saved_copy = temp_rubrics / "TEST-RUBRIC-1.json"
    assert saved_copy.exists()
    
    # Check loading
    loaded = load_rubric("TEST-RUBRIC-1")
    assert loaded.rubric_id == "TEST-RUBRIC-1"
    assert len(loaded.criteria) == 1


def test_ingest_exam_pdf(dummy_pdf_file, temp_dirs):
    temp_upload, _ = temp_dirs
    submission = ingest_exam_pdf(
        pdf_path=dummy_pdf_file,
        student_id="student_123",
        exam_id="EXAM-1",
        rubric_id="TEST-RUBRIC-1",
    )
    assert submission.student_id == "student_123"
    assert submission.exam_id == "EXAM-1"
    assert submission.status == "ingested"
    assert Path(submission.pdf_path).exists()
    assert Path(submission.pdf_path).parent == temp_upload


def test_bulk_ingest_directory(tmp_path, temp_dirs):
    # Create 3 dummy PDFs
    for name in ["stud_a.pdf", "stud_b.pdf", "stud_c.pdf"]:
        pdf_path = tmp_path / name
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

    submissions = bulk_ingest_directory(
        pdf_dir=tmp_path,
        exam_id="EXAM-BATCH-1",
        rubric_id="TEST-RUBRIC-1"
    )
    assert len(submissions) == 3
    assert {s.student_id for s in submissions} == {"stud_a", "stud_b", "stud_c"}
    for sub in submissions:
        assert sub.status == "ingested"
