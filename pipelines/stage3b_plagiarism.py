"""
GradeOps - Pipeline Stage 3b: Structural Plagiarism Detection
Maps the underlying reasoning architecture of every answer and flags
highly similar logic structures across submissions - catching collusion
even when exact phrasing and handwriting differ.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from schemas.models import ExtractedSubmission

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.82   # flag pairs above this score
MIN_CLUSTER_SIZE = 2           # minimum submissions in a plagiarism cluster


# ─── Logic Graph Representation ───────────────────────────────────────────────

@dataclass
class LogicNode:
    """A single reasoning step extracted from a student answer."""
    step_id: str
    description: str
    operation: str        # e.g. "differentiate", "substitute", "conclude"
    operands: list[str]   # generic symbols e.g. ["f", "x"]


@dataclass
class LogicGraph:
    """Directed graph of reasoning steps for one answer."""
    submission_id: str
    criterion_name: str
    nodes: list[LogicNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def fingerprint(self) -> str:
        """
        Produce a canonical fingerprint of the logic structure.
        Invariant to variable names and surface phrasing.
        """
        node_map = {node.step_id: node.operation for node in self.nodes}
        structure = sorted(
            (node.operation, len(node.operands)) for node in self.nodes
        )
        edge_structure = sorted(
            (node_map[u], node_map[v])
            for u, v in self.edges
            if u in node_map and v in node_map
        ) if self.edges else []
        payload = json.dumps(
            {"nodes": structure, "edges": edge_structure}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ─── LLM-based Logic Extraction ───────────────────────────────────────────────

EXTRACTION_SYSTEM = """You extract the logical reasoning structure from a student's
mathematical or written answer. Focus on WHAT operations are performed and in
WHAT ORDER - not the specific numbers or variable names used."""

EXTRACTION_TEMPLATE = """Extract the logic graph from this student answer for: {criterion}

STUDENT ANSWER:
{answer}

Return JSON with this exact schema:
{{
  "nodes": [
    {{"step_id": "s0", "description": "...", "operation": "...", "operands": [...]}}
  ],
  "edges": [["s0","s1"], ["s1","s2"]]
}}

Operations should be abstract verbs: differentiate, integrate, substitute, simplify,
factor, solve, conclude, define, apply_theorem, etc.
Operands should be generic symbols: f, g, x, y, expr_A, etc.
"""


# ─── Retry-wrapped LLM call ───────────────────────────────────────────────────

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _invoke_llm(llm: ChatOpenAI, messages: list) -> Any:
    """LLM call with automatic exponential-backoff retry (up to 3 attempts)."""
    return llm.invoke(messages)


def _parse_logic_graph_response(content: Any) -> tuple[list[LogicNode], list[tuple[str, str]]]:
    """Parse the LLM response, accepting raw JSON or fenced JSON."""
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
    nodes = [LogicNode(**n) for n in data.get("nodes", [])]
    edges = [tuple(e) for e in data.get("edges", [])]
    return nodes, edges


# ─── Logic Graph Extraction ───────────────────────────────────────────────────

def extract_logic_graph(
    submission_id: str,
    criterion_name: str,
    answer_text: str,
    llm: ChatOpenAI | None = None,
) -> LogicGraph:
    """
    Use an LLM to extract the underlying reasoning graph from a student answer.
    Returns an empty LogicGraph (not a crash) if the LLM call or parse fails.

    Args:
        submission_id:  Submission identifier.
        criterion_name: Which exam criterion this answer addresses.
        answer_text:    Transcribed student answer text.
        llm:            Optional pre-instantiated LLM; created if None.

    Returns:
        LogicGraph representing the answer's reasoning structure.
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = EXTRACTION_TEMPLATE.format(
        criterion=criterion_name,
        answer=answer_text,
    )
    messages = [SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=prompt)]

    # ── LLM call with retry ───────────────────────────────────────────────────
    try:
        response = _invoke_llm(llm, messages)
    except Exception as exc:
        logger.error(
            "Logic graph LLM call failed for %s / '%s' after all retries: %s",
            submission_id, criterion_name, exc,
        )
        return LogicGraph(
            submission_id=submission_id,
            criterion_name=criterion_name,
        )

    # ── JSON parse with graceful fallback ─────────────────────────────────────
    try:
        nodes, edges = _parse_logic_graph_response(response.content)
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        logger.warning(
            "Failed to parse logic graph for %s / '%s': %s | raw=%r",
            submission_id, criterion_name, exc, response.content[:200],
        )
        nodes, edges = [], []

    return LogicGraph(
        submission_id=submission_id,
        criterion_name=criterion_name,
        nodes=nodes,
        edges=edges,
    )


# ─── Similarity Scoring ───────────────────────────────────────────────────────

def _jaccard_similarity(g1: LogicGraph, g2: LogicGraph) -> float:
    """
    Compute structural similarity using Jaccard on (operation, arity) node multisets.
    """
    ops1 = Counter((n.operation, len(n.operands)) for n in g1.nodes)
    ops2 = Counter((n.operation, len(n.operands)) for n in g2.nodes)

    if not ops1 or not ops2:
        return 0.0

    intersection = sum((ops1 & ops2).values())
    union = sum((ops1 | ops2).values())
    return intersection / union


def _edge_similarity(g1: LogicGraph, g2: LogicGraph) -> float:
    """Compare edge patterns (operation-transition sequences)."""
    def edge_ops(g: LogicGraph) -> set[tuple[str, str]]:
        node_map = {n.step_id: n.operation for n in g.nodes}
        return {(node_map.get(u, u), node_map.get(v, v)) for u, v in g.edges}

    e1, e2 = edge_ops(g1), edge_ops(g2)
    if not e1 or not e2:
        return 0.0
    return len(e1 & e2) / len(e1 | e2)


def compute_structural_similarity(g1: LogicGraph, g2: LogicGraph) -> float:
    """
    Combined structural similarity score [0, 1].
    Node similarity weighted 65%, edge similarity 35%.
    """
    node_sim = _jaccard_similarity(g1, g2)
    edge_sim = _edge_similarity(g1, g2)
    return 0.65 * node_sim + 0.35 * edge_sim


# ─── Plagiarism Detection ─────────────────────────────────────────────────────

@dataclass
class PlagiarismAlert:
    """A detected cluster of suspiciously similar submissions."""
    criterion_name: str
    submission_ids: list[str]
    similarity_scores: dict[str, float]   # "sid_a:sid_b" -> score
    fingerprints: dict[str, str]           # submission_id -> logic fingerprint


def detect_plagiarism(
    submissions: list[ExtractedSubmission],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[PlagiarismAlert]:
    """
    Detect structural plagiarism across a batch of extracted submissions.

    Algorithm:
    1. Extract logic graphs for every (submission, criterion) pair.
       Failures are silently skipped (empty graphs) rather than crashing.
    2. Compare all pairs per criterion using structural similarity.
    3. Cluster submissions exceeding the similarity threshold via BFS.
    4. Return PlagiarismAlerts for clusters with at least MIN_CLUSTER_SIZE members.

    Args:
        submissions: All extracted submissions in the exam batch.
        threshold:   Similarity score above which a pair is flagged.

    Returns:
        List of PlagiarismAlert objects, one per suspicious cluster.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    alerts: list[PlagiarismAlert] = []

    # Gather all criterion names across submissions
    criterion_names: set[str] = set()
    for sub in submissions:
        for ans in sub.answers:
            criterion_names.add(ans.criterion_name)

    for criterion in criterion_names:
        graphs: dict[str, LogicGraph] = {}

        for sub in submissions:
            answer_text = next(
                (a.raw_text for a in sub.answers if a.criterion_name == criterion),
                "",
            )
            if not answer_text.strip():
                continue

            # extract_logic_graph() now returns empty graph on failure
            # instead of raising, so this loop never crashes
            graphs[sub.submission_id] = extract_logic_graph(
                sub.submission_id, criterion, answer_text, llm
            )
            if not graphs[sub.submission_id].nodes:
                logger.warning(
                    "Skipping empty logic graph for %s / '%s'",
                    sub.submission_id, criterion,
                )
                del graphs[sub.submission_id]

        if len(graphs) < 2:
            continue

        # ── Pairwise similarity ───────────────────────────────────────────────
        similarity_scores: dict[str, float] = {}
        flagged: set[str] = set()
        adjacency: dict[str, set[str]] = defaultdict(set)

        for sid_a, sid_b in combinations(graphs.keys(), 2):
            score = compute_structural_similarity(graphs[sid_a], graphs[sid_b])
            key = f"{sid_a}:{sid_b}"
            similarity_scores[key] = round(score, 4)

            if score >= threshold:
                flagged.add(sid_a)
                flagged.add(sid_b)
                adjacency[sid_a].add(sid_b)
                adjacency[sid_b].add(sid_a)
                logger.warning(
                    "Plagiarism flag [%s] %s <-> %s (score=%.3f)",
                    criterion, sid_a[:8], sid_b[:8], score,
                )

        # ── BFS clustering ────────────────────────────────────────────────────
        visited: set[str] = set()
        for start in flagged:
            if start in visited:
                continue
            cluster: list[str] = []
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                queue.extend(adjacency[node] - visited)

            if len(cluster) >= MIN_CLUSTER_SIZE:
                cluster_scores = {
                    k: v for k, v in similarity_scores.items()
                    if all(sid in cluster for sid in k.split(":", 1))
                }
                fingerprints = {sid: graphs[sid].fingerprint() for sid in cluster}
                alerts.append(PlagiarismAlert(
                    criterion_name=criterion,
                    submission_ids=cluster,
                    similarity_scores=cluster_scores,
                    fingerprints=fingerprints,
                ))

    logger.info(
        "Plagiarism detection complete: %d alert(s) across %d criteria",
        len(alerts), len(criterion_names),
    )
    return alerts


def apply_plagiarism_flags(
    grading_results: list,
    alerts: list[PlagiarismAlert],
) -> list:
    """
    Annotate GradingResult objects with plagiarism flags based on alerts.

    Args:
        grading_results: Results from Stage 3a evaluation.
        alerts:          Alerts from detect_plagiarism().

    Returns:
        Updated grading_results with plagiarism_flag and similar-to fields set.
    """
    flagged_map: dict[str, set[str]] = defaultdict(set)
    for alert in alerts:
        for sid in alert.submission_ids:
            flagged_map[sid].update(
                s for s in alert.submission_ids if s != sid
            )

    for result in grading_results:
        if result.submission_id in flagged_map:
            result.plagiarism_flag = True
            result.plagiarism_similar_to = list(flagged_map[result.submission_id])

    return grading_results
