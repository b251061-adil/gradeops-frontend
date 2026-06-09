"""
GradeOps - Pipeline Stage 3a: Agentic LLM Cognitive Engine
Uses LangChain + LangGraph to evaluate extracted answers against
JSON rubric conditions with multi-step reasoning and partial credit.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from schemas.models import (
    ConditionResult,
    CriterionResult,
    ExtractedSubmission,
    GradingResult,
    GradingRubric,
    RubricCondition,
    RubricCriterion,
    StudentSubmission,
)

logger = logging.getLogger(__name__)


# ─── LLM Setup ────────────────────────────────────────────────────────────────

def get_grading_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Return a deterministic LLM for consistent rubric evaluation.
    temperature=0 is critical for reproducible grading.
    """
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ─── LangGraph State ──────────────────────────────────────────────────────────

class GradingState(TypedDict):
    """State passed through the LangGraph grading workflow."""
    submission_id: str
    student_id: str
    rubric: dict
    answers: list[dict]           # serialised ExtractedAnswer list, keyed by criterion_name
    criterion_results: list[dict]
    current_criterion_idx: int
    total_score: float
    max_score: float
    done: bool


# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic grader. Your task is to evaluate a
student's answer against specific rubric conditions and award points objectively.

Rules:
1. Evaluate ONLY what is explicitly stated in the rubric condition.
2. Award partial credit exactly as the rubric specifies - do not invent deductions.
3. Your reasoning must cite specific parts of the student's answer.
4. Return ONLY valid JSON matching the required schema - no preamble or explanation.
5. Be consistent: identical logic always produces identical scores.
"""

CONDITION_EVAL_TEMPLATE = """Evaluate whether this rubric condition is satisfied:

RUBRIC CONDITION:
  ID: {condition_id}
  Description: {description}
  Points available: {points}
  Partial credit allowed: {is_partial}

STUDENT ANSWER:
{student_answer}

Return JSON with this exact schema:
{{
  "condition_id": "{condition_id}",
  "met": <true|false>,
  "partial": <true|false>,
  "points_awarded": <number>,
  "reasoning": "<concise explanation citing specific parts of the answer>"
}}
"""


# ─── Retry-wrapped LLM call ───────────────────────────────────────────────────

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _invoke_llm(llm: ChatOpenAI, messages: list) -> Any:
    """
    Call the LLM with automatic exponential-backoff retry.
    Retries up to 3 times on any exception (rate limits, timeouts, etc.).
    Waits 2s -> 4s -> 8s ... up to 60s between attempts.
    """
    return llm.invoke(messages)


def _parse_llm_json(content: Any) -> dict[str, Any]:
    """Parse raw or fenced JSON returned by the LLM."""
    if not isinstance(content, str):
        raise TypeError("LLM response content must be a string")

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise TypeError("LLM response JSON must be an object")
    return data


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Serialize Pydantic v2 or v1 models to a plain dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError(f"Object of type {type(model).__name__} is not serializable")


# ─── Node Functions ───────────────────────────────────────────────────────────

def _evaluate_condition(
    llm: ChatOpenAI,
    condition: RubricCondition,
    student_answer: str,
) -> ConditionResult:
    """
    Call the LLM to evaluate a single rubric condition.
    Returns a zero-score fallback result if the LLM response cannot be parsed.
    """
    if not student_answer.strip():
        return ConditionResult(
            condition_id=condition.condition_id,
            met=False,
            partial=False,
            points_awarded=0.0,
            reasoning="No answer provided for this criterion.",
        )

    prompt = CONDITION_EVAL_TEMPLATE.format(
        condition_id=condition.condition_id,
        description=condition.description,
        points=condition.points,
        is_partial=condition.is_partial_credit,
        student_answer=student_answer,
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    try:
        response = _invoke_llm(llm, messages)
    except Exception as exc:
        # All 3 retries exhausted - return a safe zero-score result
        logger.error(
            "LLM call failed for condition %s after all retries: %s",
            condition.condition_id, exc,
        )
        return ConditionResult(
            condition_id=condition.condition_id,
            met=False,
            partial=False,
            points_awarded=0.0,
            reasoning=f"llm_call_error: {exc}",
        )

    # Parse JSON - if it fails, return a zero-score fallback instead of crashing
    try:
        data = _parse_llm_json(response.content)
        data["condition_id"] = condition.condition_id
        result = ConditionResult(**data)
        result.points_awarded = max(0.0, min(float(result.points_awarded), condition.points))
        if not condition.is_partial_credit and not result.met:
            result.partial = False
            result.points_awarded = 0.0
        return result
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not parse LLM response for condition %s: %s | raw=%r",
            condition.condition_id, exc, response.content[:200],
        )
        return ConditionResult(
            condition_id=condition.condition_id,
            met=False,
            partial=False,
            points_awarded=0.0,
            reasoning=f"llm_parse_error: {exc}",
        )


def _evaluate_criterion(
    llm: ChatOpenAI,
    criterion: RubricCriterion,
    student_answer: str,
) -> CriterionResult:
    """Evaluate all conditions within one rubric criterion."""
    condition_results: list[ConditionResult] = []
    total_awarded = 0.0

    for condition in criterion.conditions:
        result = _evaluate_condition(llm, condition, student_answer)
        condition_results.append(result)
        total_awarded += result.points_awarded

    # Clamp criterion total
    total_awarded = max(0.0, min(total_awarded, criterion.max_points))

    # Build structured justification
    justification_parts = [
        f"[{r.condition_id}] {'met' if r.met else 'not met'} {r.reasoning}"
        for r in condition_results
    ]
    justification = " | ".join(justification_parts)

    return CriterionResult(
        criterion_name=criterion.name,
        points_awarded=total_awarded,
        max_points=criterion.max_points,
        condition_results=condition_results,
        justification=justification,
    )


# ─── LangGraph Nodes ──────────────────────────────────────────────────────────

def grade_next_criterion(state: GradingState) -> GradingState:
    """LangGraph node: grade one criterion and advance the index."""
    llm = get_grading_llm()
    rubric = GradingRubric(**state["rubric"])
    idx = state["current_criterion_idx"]

    if idx >= len(rubric.criteria):
        return {**state, "done": True}

    criterion = rubric.criteria[idx]

    # ── Name-based lookup (fixed from broken index-based lookup) ──────────────
    # answers list now contains dicts with criterion_name set by stage 2
    answers_by_name: dict[str, str] = {
        a["criterion_name"]: a.get("raw_text", "")
        for a in state["answers"]
    }
    answer_text = answers_by_name.get(criterion.name, "")

    if not answer_text:
        logger.warning(
            "No answer found for criterion '%s' in submission %s - awarding 0",
            criterion.name, state["submission_id"],
        )

    logger.info(
        "Grading criterion %d/%d: '%s' (%d chars)",
        idx + 1, len(rubric.criteria), criterion.name, len(answer_text),
    )

    result = _evaluate_criterion(llm, criterion, answer_text)

    criterion_results = state["criterion_results"] + [_model_to_dict(result)]
    total_score = state["total_score"] + result.points_awarded
    max_score = state["max_score"] + result.max_points

    return {
        **state,
        "criterion_results": criterion_results,
        "current_criterion_idx": idx + 1,
        "total_score": total_score,
        "max_score": max_score,
        "done": (idx + 1) >= len(rubric.criteria),
    }


def should_continue(state: GradingState) -> str:
    return END if state["done"] else "grade_next_criterion"


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_grading_graph() -> Any:
    """Build and compile the LangGraph grading workflow."""
    graph = StateGraph(GradingState)
    graph.add_node("grade_next_criterion", grade_next_criterion)
    graph.set_entry_point("grade_next_criterion")
    graph.add_conditional_edges(
        "grade_next_criterion",
        should_continue,
        {"grade_next_criterion": "grade_next_criterion", END: END},
    )
    return graph.compile()


# ─── Main Evaluation Entry Point ─────────────────────────────────────────────

def evaluate_submission(
    extracted: ExtractedSubmission,
    rubric: GradingRubric,
) -> GradingResult:
    """
    Run the full agentic evaluation pipeline for one submission.

    Args:
        extracted:  Transcribed answers from Stage 2.
        rubric:     Validated rubric from Stage 1.

    Returns:
        GradingResult with AI-proposed scores and justifications.
    """
    app = build_grading_graph()

    initial_state: GradingState = {
        "submission_id": extracted.submission_id,
        "student_id": extracted.student_id,
        "rubric": _model_to_dict(rubric),
        "answers": [_model_to_dict(a) for a in extracted.answers],
        "criterion_results": [],
        "current_criterion_idx": 0,
        "total_score": 0.0,
        "max_score": 0.0,
        "done": False,
    }

    final_state = app.invoke(initial_state)
    criterion_results = [CriterionResult(**r) for r in final_state["criterion_results"]]

    return GradingResult(
        submission_id=extracted.submission_id,
        student_id=extracted.student_id,
        rubric_id=rubric.rubric_id,
        total_score=final_state["total_score"],
        max_score=final_state["max_score"],
        criterion_results=criterion_results,
        evaluation_model="gpt-4o",
        status="proposed",
    )


# ─── Batch Evaluation ─────────────────────────────────────────────────────────

def batch_evaluate(
    extracted_list: list[ExtractedSubmission],
    rubric: GradingRubric,
    submissions: Optional[list[StudentSubmission]] = None,
) -> list[GradingResult]:
    """
    Evaluate a batch of extracted submissions against the same rubric.

    Args:
        extracted_list: Transcribed submissions from Stage 2.
        rubric:         Validated rubric from Stage 1.
        submissions:    Optional original StudentSubmission objects.
                        If provided, their .status field is updated to
                        'evaluated' after successful grading.

    Returns:
        List of GradingResult objects (one per successfully evaluated submission).
    """
    # Build a lookup so we can update status without a second loop
    sub_map: dict[str, StudentSubmission] = {}
    if submissions:
        sub_map = {s.submission_id: s for s in submissions}

    results = []
    for extracted in extracted_list:
        try:
            result = evaluate_submission(extracted, rubric)
            results.append(result)

            # Update submission status if caller provided the original objects
            if extracted.submission_id in sub_map:
                sub_map[extracted.submission_id].status = "evaluated"

            logger.info(
                "Evaluated %s: %.1f/%.1f",
                extracted.submission_id,
                result.total_score,
                result.max_score,
            )
        except Exception as exc:
            logger.error(
                "Evaluation failed for %s: %s",
                extracted.submission_id, exc,
            )

    return results
