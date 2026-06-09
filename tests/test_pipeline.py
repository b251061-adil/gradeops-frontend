import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from reportlab.pdfgen import canvas

from pipelines.stage1_ingestion import ingest_rubric, ingest_exam_pdf, UPLOAD_DIR, RUBRIC_DIR
from pipelines.stage2_extraction import extract_submission
from pipelines.stage3a_evaluation import evaluate_submission
from orchestrator import run_pipeline, PipelineConfig
from schemas.models import ExtractedSubmission, ExtractedAnswer, GradingRubric, RubricCriterion


@pytest.fixture
def temp_stores(tmp_path, monkeypatch):
    temp_upload = tmp_path / "uploads"
    temp_rubrics = tmp_path / "rubrics"
    temp_upload.mkdir()
    temp_rubrics.mkdir()
    
    monkeypatch.setattr("pipelines.stage1_ingestion.UPLOAD_DIR", temp_upload)
    monkeypatch.setattr("pipelines.stage1_ingestion.RUBRIC_DIR", temp_rubrics)
    return temp_upload, temp_rubrics


def create_minimal_pdf(path: Path):
    c = canvas.Canvas(str(path))
    c.drawString(100, 750, "Student answer: Problem 1 is 42. Problem 2: f'(x) = 3x^2 - 6x - 9.")
    c.showPage()
    c.save()


def test_ingest_rubric(temp_stores):
    # Load example_rubric.json from the project root
    rubric_path = Path("example_rubric.json")
    rubric = ingest_rubric(rubric_path)
    assert rubric.rubric_id == "MTH-101-A"
    assert len(rubric.criteria) == 2


def test_ingest_pdf(tmp_path, temp_stores):
    pdf_path = tmp_path / "student_1.pdf"
    create_minimal_pdf(pdf_path)
    
    submission = ingest_exam_pdf(
        pdf_path=pdf_path,
        student_id="student_999",
        exam_id="EXAM-999",
        rubric_id="MTH-101-A"
    )
    
    assert submission.status == "ingested"
    assert Path(submission.pdf_path).exists()


@patch("pipelines.stage2_extraction._extract_with_qwen_vl")
def test_extract_submission(mock_qwen_vl, tmp_path, temp_stores):
    pdf_path = tmp_path / "student_1.pdf"
    create_minimal_pdf(pdf_path)
    
    submission = ingest_exam_pdf(
        pdf_path=pdf_path,
        student_id="student_999",
        exam_id="EXAM-999",
        rubric_id="MTH-101-A"
    )
    
    # Mock VLM extraction response
    mock_qwen_vl.return_value = [
        "Q1: The answer is 42. Q2: Because f(x)=x^2."
    ]
    
    # We mock segment_answers_by_criterion to avoid LLM calls
    with patch("pipelines.stage2_extraction.segment_answers_by_criterion") as mock_segment:
        mock_segment.return_value = {
            "Problem 1": "Q1: The answer is 42.",
            "Problem 2": "Q2: Because f(x)=x^2."
        }
        
        criteria = [
            RubricCriterion(name="Problem 1", max_points=10, description="desc1"),
            RubricCriterion(name="Problem 2", max_points=15, description="desc2")
        ]
        
        extracted = extract_submission(submission, backend="qwen-vl", rubric_criteria=criteria)
        
        assert extracted.submission_id == submission.submission_id
        assert len(extracted.answers) == 2
        assert extracted.answers[0].criterion_name == "Problem 1"
        assert extracted.answers[0].raw_text == "Q1: The answer is 42."
        assert extracted.answers[1].criterion_name == "Problem 2"
        assert extracted.answers[1].raw_text == "Q2: Because f(x)=x^2."


@patch("pipelines.stage3a_evaluation.get_grading_llm")
def test_evaluate_submission(mock_get_llm, temp_stores):
    # Load example rubric
    rubric_path = Path("example_rubric.json")
    with open(rubric_path) as f:
        rubric = GradingRubric(**json.load(f))
        
    extracted = ExtractedSubmission(
        submission_id="sub_123",
        student_id="student_123",
        rubric_id="MTH-101-A",
        answers=[
            ExtractedAnswer(criterion_name="Problem 1", raw_text="Answer to P1 is 42", confidence=0.9, page_number=1),
            ExtractedAnswer(criterion_name="Problem 2", raw_text="f'(x) = 3x^2 - 6x - 9", confidence=0.9, page_number=1)
        ],
        extraction_model="Qwen-VL-Chat"
    )
    
    # Mock Evaluation LLM (using new _invoke_llm implementation)
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    # We need responses for each condition inside the example rubric
    # Problem 1 has 3 conditions, Problem 2 has 4 conditions = 7 calls in total
    mock_responses = []
    
    # Problem 1 condition mock responses
    p1_conds = ["P1_C1", "P1_C2", "P1_C3"]
    for cid in p1_conds:
        resp = MagicMock()
        resp.content = json.dumps({
            "condition_id": cid,
            "met": True,
            "partial": False,
            "points_awarded": 3.0,
            "reasoning": "Criterion condition satisfied"
        })
        mock_responses.append(resp)
        
    # Problem 2 condition mock responses
    p2_conds = ["P2_C1", "P2_C2", "P2_C3", "P2_C4"]
    for cid in p2_conds:
        resp = MagicMock()
        resp.content = json.dumps({
            "condition_id": cid,
            "met": True,
            "partial": False,
            "points_awarded": 3.0,
            "reasoning": "Criterion condition satisfied"
        })
        mock_responses.append(resp)
        
    mock_llm.invoke.side_effect = mock_responses
    
    result = evaluate_submission(extracted, rubric, exam_id="EXAM-1")
    assert result.submission_id == "sub_123"
    assert result.total_score >= 0
    assert result.status == "proposed"
    for cr in result.criterion_results:
        assert cr.justification != ""


@patch("pipelines.stage2_extraction.segment_answers_by_criterion")
@patch("pipelines.stage2_extraction._extract_with_qwen_vl")
@patch("pipelines.stage3a_evaluation.get_grading_llm")
@patch("pipelines.stage3b_plagiarism.ChatOpenAI")
@patch("httpx.post")
def test_full_pipeline(mock_post, mock_plag_chat, mock_get_llm, mock_qwen_vl, mock_segment, tmp_path, temp_stores):
    # Setup temp PDF dir
    pdf_dir = tmp_path / "scans"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "student_1.pdf"
    create_minimal_pdf(pdf_path)
    
    # Mock extraction
    mock_segment.return_value = {
        "Problem 1": "Q1: The answer is 42.",
        "Problem 2": "Q2: Because f(x)=x^2."
    }
    mock_qwen_vl.return_value = ["Q1: The answer is 42. Q2: f(x)=x^2"]
    
    # Mock evaluation
    mock_eval_llm = MagicMock()
    mock_get_llm.return_value = mock_eval_llm
    
    resp_eval = MagicMock()
    resp_eval.content = json.dumps({
        "condition_id": "P1_C1",
        "met": True,
        "partial": False,
        "points_awarded": 3.0,
        "reasoning": "Satisfied"
    })
    mock_eval_llm.invoke.return_value = resp_eval
    
    # Mock plagiarism
    mock_plag_llm = MagicMock()
    mock_plag_chat.return_value = mock_plag_llm
    resp_plag = MagicMock()
    resp_plag.content = json.dumps({
        "nodes": [
            {"step_id": "s0", "description": "diff", "operation": "differentiate", "operands": ["x"]},
            {"step_id": "s1", "description": "solve", "operation": "solve", "operands": ["y"]}
        ],
        "edges": [["s0", "s1"]]
    })
    mock_plag_llm.invoke.return_value = resp_plag
    
    # Mock dashboard loader
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {"loaded": 1}
    mock_post.return_value = mock_post_resp
    
    config = PipelineConfig(
        rubric_json_path=Path("example_rubric.json"),
        exam_pdf_dir=pdf_dir,
        exam_id="MTH-101-A",
        vlm_backend="qwen-vl",
    )
    
    output = run_pipeline(config)
    assert len(output.grading_results) > 0
    assert output.stats["evaluated"] > 0
