import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import fitz

from orchestrator import PipelineConfig, run_pipeline


@pytest.fixture
def dummy_environment(tmp_path, monkeypatch):
    # Setup directories
    pdf_dir = tmp_path / "scans"
    pdf_dir.mkdir()
    
    # Save a dummy PDF
    pdf_path = pdf_dir / "student_1.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    
    # Save a dummy rubric
    rubric_data = {
        "rubric_id": "TEST-RUBRIC",
        "course_id": "CS101",
        "exam_name": "Final",
        "total_points": 10.0,
        "criteria": [
            {
                "name": "Problem 1",
                "max_points": 10.0,
                "description": "Solve P1",
                "conditions": [
                    {
                        "condition_id": "P1_C1",
                        "description": "Correct working",
                        "points": 10.0,
                        "is_partial_credit": True
                    }
                ]
            }
        ]
    }
    rubric_path = tmp_path / "rubric.json"
    with open(rubric_path, "w") as f:
        json.dump(rubric_data, f)
        
    # Redirect upload stores to temp dirs
    temp_upload = tmp_path / "uploads"
    temp_rubrics = tmp_path / "rubrics"
    temp_upload.mkdir()
    temp_rubrics.mkdir()
    
    monkeypatch.setattr("pipelines.stage1_ingestion.UPLOAD_DIR", temp_upload)
    monkeypatch.setattr("pipelines.stage1_ingestion.RUBRIC_DIR", temp_rubrics)
    
    return pdf_dir, rubric_path


@patch("pipelines.stage2_extraction.segment_answers_by_criterion")
@patch("pipelines.stage3a_evaluation.get_grading_llm")
@patch("pipelines.stage3b_plagiarism.ChatOpenAI")
@patch("httpx.post")
def test_full_pipeline_run(mock_httpx_post, mock_plagiarism_chat, mock_evaluation_llm_getter, mock_segment, dummy_environment):
    pdf_dir, rubric_path = dummy_environment
    
    # Mock Stage 2 (Criterion segmenting LLM)
    mock_segment.return_value = {
        "Problem 1": "Problem 1: To evaluate the integral of x * ln(x) dx, we use integration by parts (IBP).\nLet u = ln(x) => du = 1/x dx.\nLet dv = x dx => v = x^2 / 2.\nThen integral = u*v - integral(v * du) = (ln(x) * x^2)/2 - integral(x^2/2 * 1/x dx)\n= (x^2 * ln(x))/2 - 1/2 * integral(x dx) = (x^2 * ln(x))/2 - x^2/4 + C.\nThe final answer is (x^2 * ln(x))/2 - x^2/4 + C."
    }
    
    # Mock Stage 3a (Evaluation LLM)
    mock_eval_llm = MagicMock()
    mock_evaluation_llm_getter.return_value = mock_eval_llm
    
    resp_eval = MagicMock()
    resp_eval.content = json.dumps({
        "condition_id": "P1_C1",
        "met": True,
        "partial": True,
        "points_awarded": 8.0,
        "reasoning": "Almost perfect work."
    })
    mock_eval_llm.invoke.return_value = resp_eval
    
    # Mock Stage 3b (Plagiarism LLM)
    mock_plag_llm = MagicMock()
    mock_plagiarism_chat.return_value = mock_plag_llm
    
    resp_plag = MagicMock()
    resp_plag.content = json.dumps({
        "nodes": [
            {"step_id": "s0", "description": "diff", "operation": "differentiate", "operands": ["x"]}
        ],
        "edges": []
    })
    mock_plag_llm.invoke.return_value = resp_plag
    
    # Mock Stage 4 (HTTP Client upload to Dashboard)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"loaded": 1}
    mock_httpx_post.return_value = mock_response
    
    config = PipelineConfig(
        rubric_json_path=rubric_path,
        exam_pdf_dir=pdf_dir,
        exam_id="EXAM-TEST-100",
        vlm_backend="mock",
    )
    
    output = run_pipeline(config)
    
    assert output.stats["total_submissions"] == 1
    assert output.stats["extracted"] == 1
    assert output.stats["evaluated"] == 1
    assert output.stats["average_score"] == 8.0
    assert output.stats["max_score"] == 10.0
    
    # Verify dashboard load post request was sent
    assert mock_httpx_post.called
    posted_json = mock_httpx_post.call_args[1]["json"]
    assert "results" in posted_json
    assert len(posted_json["results"]) == 1
    assert posted_json["results"][0]["submission_id"] == output.grading_results[0].submission_id
