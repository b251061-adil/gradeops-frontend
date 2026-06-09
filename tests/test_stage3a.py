import json
import pytest
from unittest.mock import MagicMock, patch

from pipelines.stage3a_evaluation import (
    evaluate_submission,
    batch_evaluate,
)
from schemas.models import (
    ExtractedSubmission,
    ExtractedAnswer,
    GradingRubric,
    RubricCriterion,
    RubricCondition,
    StudentSubmission,
)


@pytest.fixture
def dummy_rubric():
    return GradingRubric(
        rubric_id="RUBRIC-1",
        course_id="CS101",
        exam_name="Exam 1",
        total_points=10.0,
        criteria=[
            RubricCriterion(
                name="Problem 1",
                max_points=10.0,
                description="desc",
                conditions=[
                    RubricCondition(
                        condition_id="P1_C1",
                        description="Method",
                        points=4.0,
                        is_partial_credit=False,
                    ),
                    RubricCondition(
                        condition_id="P1_C2",
                        description="Calculation",
                        points=6.0,
                        is_partial_credit=True,
                    ),
                ],
            )
        ],
    )


@pytest.fixture
def dummy_extracted():
    return ExtractedSubmission(
        submission_id="sub_123",
        student_id="student_123",
        rubric_id="RUBRIC-1",
        answers=[
            ExtractedAnswer(
                criterion_name="Problem 1",
                raw_text="The answer is correct.",
                confidence=0.9,
                page_number=1,
            )
        ],
        extraction_model="MockVLM",
    )


@pytest.fixture
def dummy_submission():
    return StudentSubmission(
        submission_id="sub_123",
        student_id="student_123",
        exam_id="EXAM-123",
        rubric_id="RUBRIC-1",
        pdf_path="storage/uploads/sub_123.pdf",
        status="extracted",
    )


@patch("pipelines.stage3a_evaluation.get_grading_llm")
def test_evaluate_submission(mock_get_llm, dummy_extracted, dummy_rubric):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    # Mock two LLM calls for two conditions
    resp1 = MagicMock()
    resp1.content = json.dumps({
        "condition_id": "P1_C1",
        "met": True,
        "partial": False,
        "points_awarded": 4.0,
        "reasoning": "Correct method used."
    })
    
    resp2 = MagicMock()
    resp2.content = json.dumps({
        "condition_id": "P1_C2",
        "met": True,
        "partial": True,
        "points_awarded": 3.0,
        "reasoning": "Minor arithmetic error."
    })
    
    mock_llm.invoke.side_effect = [resp1, resp2]

    result = evaluate_submission(dummy_extracted, dummy_rubric, exam_id="EXAM-123")
    assert result.submission_id == "sub_123"
    assert result.exam_id == "EXAM-123"
    assert result.total_score == 7.0
    assert result.max_score == 10.0
    assert len(result.criterion_results) == 1
    assert len(result.criterion_results[0].condition_results) == 2
    assert result.criterion_results[0].points_awarded == 7.0


@patch("pipelines.stage3a_evaluation.get_grading_llm")
def test_batch_evaluate(mock_get_llm, dummy_extracted, dummy_rubric, dummy_submission):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    resp1 = MagicMock()
    resp1.content = json.dumps({
        "condition_id": "P1_C1",
        "met": True,
        "partial": False,
        "points_awarded": 4.0,
        "reasoning": "Correct."
    })
    resp2 = MagicMock()
    resp2.content = json.dumps({
        "condition_id": "P1_C2",
        "met": False,
        "partial": False,
        "points_awarded": 0.0,
        "reasoning": "Missing calculation."
    })
    mock_llm.invoke.side_effect = [resp1, resp2]

    results = batch_evaluate(
        extracted_list=[dummy_extracted],
        rubric=dummy_rubric,
        submissions=[dummy_submission]
    )

    assert len(results) == 1
    assert results[0].total_score == 4.0
    assert results[0].exam_id == "EXAM-123"
    assert dummy_submission.status == "evaluated"
