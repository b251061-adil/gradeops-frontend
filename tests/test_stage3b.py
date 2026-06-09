import json
import pytest
from unittest.mock import MagicMock, patch

from pipelines.stage3b_plagiarism import (
    LogicNode,
    LogicGraph,
    _jaccard_similarity,
    _edge_similarity,
    compute_structural_similarity,
    extract_logic_graph,
    detect_plagiarism,
    apply_plagiarism_flags,
)
from schemas.models import ExtractedSubmission, ExtractedAnswer, GradingResult


def test_graph_fingerprint():
    # Identical structure should have identical fingerprint
    g1 = LogicGraph(
        submission_id="sub_a",
        criterion_name="Problem 1",
        nodes=[
            LogicNode(step_id="s0", description="diff", operation="differentiate", operands=["x"]),
            LogicNode(step_id="s1", description="solve", operation="solve", operands=["y"])
        ],
        edges=[("s0", "s1")]
    )
    
    g2 = LogicGraph(
        submission_id="sub_b",
        criterion_name="Problem 1",
        nodes=[
            # Nodes out of order but identical structure
            LogicNode(step_id="s1", description="solve", operation="solve", operands=["z"]),
            LogicNode(step_id="s0", description="diff", operation="differentiate", operands=["a"])
        ],
        edges=[("s0", "s1")]
    )
    assert g1.fingerprint() == g2.fingerprint()


def test_similarities():
    g1 = LogicGraph(
        submission_id="sub_a",
        criterion_name="P1",
        nodes=[
            LogicNode(step_id="s0", description="d1", operation="differentiate", operands=["x"]),
            LogicNode(step_id="s1", description="s1", operation="substitute", operands=["y"])
        ],
        edges=[("s0", "s1")]
    )
    g2 = LogicGraph(
        submission_id="sub_b",
        criterion_name="P1",
        nodes=[
            LogicNode(step_id="s0", description="d2", operation="differentiate", operands=["a"]),
            LogicNode(step_id="s1", description="s2", operation="substitute", operands=["b"])
        ],
        edges=[("s0", "s1")]
    )
    
    jaccard = _jaccard_similarity(g1, g2)
    assert jaccard == 1.0
    
    edge_sim = _edge_similarity(g1, g2)
    assert edge_sim == 1.0
    
    assert compute_structural_similarity(g1, g2) == 1.0


@patch("pipelines.stage3b_plagiarism.ChatOpenAI")
def test_extract_logic_graph(mock_chat_class):
    mock_llm = MagicMock()
    mock_chat_class.return_value = mock_llm
    
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "nodes": [
            {"step_id": "s0", "description": "diff", "operation": "differentiate", "operands": ["x"]}
        ],
        "edges": []
    })
    mock_llm.invoke.return_value = mock_resp

    graph = extract_logic_graph("sub_a", "P1", "student answer", mock_llm)
    assert graph.submission_id == "sub_a"
    assert len(graph.nodes) == 1
    assert graph.nodes[0].operation == "differentiate"


@patch("pipelines.stage3b_plagiarism.ChatOpenAI")
def test_detect_plagiarism(mock_chat_class):
    mock_llm = MagicMock()
    mock_chat_class.return_value = mock_llm

    # Mock response for three submissions
    # Sub A and Sub B are structurally identical
    resp_a = MagicMock(content=json.dumps({
        "nodes": [
            {"step_id": "s0", "description": "diff", "operation": "differentiate", "operands": ["x"]}
        ],
        "edges": []
    }))
    
    resp_b = MagicMock(content=json.dumps({
        "nodes": [
            {"step_id": "s0", "description": "diff", "operation": "differentiate", "operands": ["a"]}
        ],
        "edges": []
    }))
    
    # Sub C is different
    resp_c = MagicMock(content=json.dumps({
        "nodes": [
            {"step_id": "s0", "description": "int", "operation": "integrate", "operands": ["y"]}
        ],
        "edges": []
    }))

    mock_llm.invoke.side_effect = [resp_a, resp_b, resp_c]

    subs = [
        ExtractedSubmission(
            submission_id="sub_a", student_id="std_a", rubric_id="R1",
            answers=[ExtractedAnswer(criterion_name="P1", raw_text="Answer A", confidence=0.9, page_number=1)],
            extraction_model="MockVLM"
        ),
        ExtractedSubmission(
            submission_id="sub_b", student_id="std_b", rubric_id="R1",
            answers=[ExtractedAnswer(criterion_name="P1", raw_text="Answer B", confidence=0.9, page_number=1)],
            extraction_model="MockVLM"
        ),
        ExtractedSubmission(
            submission_id="sub_c", student_id="std_c", rubric_id="R1",
            answers=[ExtractedAnswer(criterion_name="P1", raw_text="Answer C", confidence=0.9, page_number=1)],
            extraction_model="MockVLM"
        ),
    ]

    alerts = detect_plagiarism(subs, threshold=0.8)
    assert len(alerts) == 1
    assert alerts[0].criterion_name == "P1"
    assert set(alerts[0].submission_ids) == {"sub_a", "sub_b"}


def test_apply_plagiarism_flags():
    grading_results = [
        GradingResult(
            submission_id="sub_a", student_id="std_a", rubric_id="R1",
            total_score=5, max_score=10, criterion_results=[], evaluation_model="test", status="proposed"
        ),
        GradingResult(
            submission_id="sub_b", student_id="std_b", rubric_id="R1",
            total_score=5, max_score=10, criterion_results=[], evaluation_model="test", status="proposed"
        ),
        GradingResult(
            submission_id="sub_c", student_id="std_c", rubric_id="R1",
            total_score=8, max_score=10, criterion_results=[], evaluation_model="test", status="proposed"
        ),
    ]
    from pipelines.stage3b_plagiarism import PlagiarismAlert
    alerts = [
        PlagiarismAlert(
            criterion_name="P1",
            submission_ids=["sub_a", "sub_b"],
            similarity_scores={"sub_a:sub_b": 1.0},
            fingerprints={"sub_a": "fp1", "sub_b": "fp1"}
        )
    ]
    
    updated = apply_plagiarism_flags(grading_results, alerts)
    assert updated[0].plagiarism_flag is True
    assert updated[0].plagiarism_similar_to == ["sub_b"]
    assert updated[1].plagiarism_flag is True
    assert updated[1].plagiarism_similar_to == ["sub_a"]
    assert updated[2].plagiarism_flag is False
