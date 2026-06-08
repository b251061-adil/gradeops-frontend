"""
GradeOps – Pipeline Stage 3b: Structural Plagiarism Detection
Maps the underlying reasoning architecture of every answer and flags
highly similar logic structures across submissions – catching collusion
even when exact phrasing and handwriting differ.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from schemas.models import ExtractedSubmission

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.82   # flag pairs above this cosine-like score
MIN_CLUSTER_SIZE = 2           # minimum submissions in a plagiarism cluster


# ─── Logic Graph Representation ───────────────────────────────────────────────

@dataclass
class LogicNode:
    """A single reasoning step extracted from a student answer."""
    step_id: str
    description: str          # normalised description of the step
    operation: str            # e.g. "differentiate", "substitute", "conclude"
    operands: list[str]       # symbols/variables involved


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
        # Sorted sequence of (operation, arity) tuples
        structure = sorted(
            (node.operation, len(node.operands)) for node in self.nodes
        )
        edge_structure = sorted(
            (self.nodes[int(u.split("_")[-1])].operation,
             self.nodes[int(v.split("_")[-1])].operation)
            for u, v in self.edges
            if u.split("_")[-1].isdigit() and v.split("_")[-1].isdigit()
        ) if self.edges else []
        payload = json.dumps({"nodes": structure, "edges": edge_structure}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ─── LLM-based Logic Extraction ───────────────────────────────────────────────

EXTRACTION_SYSTEM = """You extract the logical reasoning structure from a student's 
mathematical or written answer. Focus on WHAT operations are performed and in 
WHAT ORDER – not the specific numbers or variable names used."""

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


def extract_logic_graph(
    submission_id: str,
    criterion_name: str,
    answer_text: str,
    llm: ChatOpenAI | None = None,
) -> LogicGraph:
    """
    Use an LLM to extract the underlying reasoning graph from a student answer.

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
    response = llm.invoke(messages)

    try:
        data = json.loads(response.content)
        nodes = [LogicNode(**n) for n in data.get("nodes", [])]
        edges = [tuple(e) for e in data.get("edges", [])]
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse logic graph for %s – using empty graph", submission_id)
        nodes, edges = [], []

    return LogicGraph(
        submission_id=submission_id,
        criterion_name=criterion_name,
        nodes=nodes,
        edges=edges,
    )


# ─── Similarity Scoring ────────────────────────────────────────────────────────

def _jaccard_similarity(g1: LogicGraph, g2: LogicGraph) -> float:
    """
    Compute structural similarity between two logic graphs using
    Jaccard similarity on their (operation, arity) node multisets.
    """
    ops1 = [(n.operation, len(n.operands)) for n in g1.nodes]
    ops2 = [(n.operation, len(n.operands)) for n in g2.nodes]

    set1, set2 = set(ops1), set(ops2)
    if not set1 and not set2:
        return 1.0  # both empty → identical
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def _edge_similarity(g1: LogicGraph, g2: LogicGraph) -> float:
    """Compare edge patterns (transition sequences)."""
    def edge_ops(g: LogicGraph) -> set[tuple[str, str]]:
        node_map = {n.step_id: n.operation for n in g.nodes}
        return {(node_map.get(u, u), node_map.get(v, v)) for u, v in g.edges}

    e1, e2 = edge_ops(g1), edge_ops(g2)
    if not e1 and not e2:
        return 1.0
    if not e1 or not e2:
        return 0.0
    return len(e1 & e2) / len(e1 | e2)


def compute_structural_similarity(g1: LogicGraph, g2: LogicGraph) -> float:
    """
    Combined structural similarity score [0, 1].
    Weights node similarity higher than edge similarity.
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
    similarity_scores: dict[str, float]   # "sid_a:sid_b" → score
    fingerprints: dict[str, str]           # submission_id → logic fingerprint


def detect_plagiarism(
    submissions: list[ExtractedSubmission],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[PlagiarismAlert]:
    """
    Detect structural plagiarism across a batch of extracted submissions.

    Algorithm:
    1. Extract logic graphs for every (submission, criterion) pair.
    2. Compare all pairs per criterion using structural similarity.
    3. Cluster submissions that exceed the similarity threshold.
    4. Return PlagiarismAlerts for clusters with ≥2 members.

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
        # Extract logic graphs for this criterion
        graphs: dict[str, LogicGraph] = {}
        for sub in submissions:
            answer_text = next(
                (a.raw_text for a in sub.answers if a.criterion_name == criterion), ""
            )
            if not answer_text.strip():
                continue
            graphs[sub.submission_id] = extract_logic_graph(
                sub.submission_id, criterion, answer_text, llm
            )

        if len(graphs) < 2:
            continue

        # Pairwise similarity
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
                    "Plagiarism flag [%s] %s ↔ %s (score=%.3f)",
                    criterion, sid_a[:8], sid_b[:8], score,
                )

        # Build clusters via BFS
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
                    if any(sid in k for sid in cluster)
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
    grading_results: list,          # list[GradingResult]
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
