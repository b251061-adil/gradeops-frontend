import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RUBRIC_PATH = BASE_DIR / 'data' / 'sample_rubric.json'

DEFAULT_RUBRIC = [
    {
        'question': 'Q1',
        'topic': 'Time Complexity',
        'max_marks': 10,
        'criteria': 'Award full marks for correct Master Theorem application. Deduct 3 marks for missing edge cases.'
    },
    {
        'question': 'Q2',
        'topic': 'Binary Search Tree',
        'max_marks': 10,
        'criteria': 'Award 5 marks for correct BST traversal and 5 marks for correct node deletion reasoning.'
    },
    {
        'question': 'Q3',
        'topic': 'Dynamic Programming',
        'max_marks': 15,
        'criteria': 'Award 5 marks for correct subproblem identification, 5 for recurrence relation, 5 for final solution.'
    },
    {
        'question': 'Q4',
        'topic': 'Graph Traversal',
        'max_marks': 10,
        'criteria': 'Award full marks for correct BFS/DFS choice, edge-case handling, and traversal proof.'
    },
    {
        'question': 'Q5',
        'topic': 'Bonus',
        'max_marks': 5,
        'criteria': 'Award partial credit for a valid bonus insight and extra marks for strong justification.'
    }
]

QUALITY_LABELS = ['perfect', 'strong', 'partial', 'weak', 'incorrect']
SAMPLE_NAMES = [
    'Aryan Sharma', 'Priya Nair', 'Rohan Mehta', 'Sneha Iyer', 'Vikram Das', 'Ananya Singh', 'Kabir Joshi'
]
SAMPLE_ROLLS = [
    'CS21B043', 'CS21B017', 'CS21B029', 'CS21B051', 'CS21B008', 'CS21B062', 'CS21B034'
]


def normalize_text(text: str) -> str:
    text = (text or '').lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_rubric() -> List[Dict[str, Any]]:
    if SAMPLE_RUBRIC_PATH.exists():
        try:
            return json.loads(SAMPLE_RUBRIC_PATH.read_text())
        except Exception:
            pass
    return DEFAULT_RUBRIC


def keyword_score(answer_text: str, rubric_item: Dict[str, Any]) -> int:
    answer_tokens = set(normalize_text(answer_text).split())
    rubric_tokens = set(normalize_text(f"{rubric_item['question']} {rubric_item['topic']} {rubric_item['criteria']}").split())
    if not answer_tokens or not rubric_tokens:
        return 0

    overlap = len(answer_tokens & rubric_tokens)
    ratio = overlap / max(1, len(rubric_tokens))
    score = int(round(ratio * rubric_item['max_marks'] * 1.2))
    return max(0, min(rubric_item['max_marks'], score))


def generate_feedback(score: int, max_marks: int) -> str:
    fraction = score / max(1, max_marks)
    if fraction >= 0.9:
        return 'Excellent solution. The answer satisfies the rubric and handles required edge cases.'
    if fraction >= 0.7:
        return 'Strong answer with minor issues. Check the final step for accuracy and completeness.'
    if fraction >= 0.45:
        return 'Partial credit awarded. The main approach is valid, but some key details are missing.'
    if fraction >= 0.2:
        return 'Attempted answer, but several rubric elements are absent or incorrect.'
    return 'Answer does not satisfy the rubric. Revisit the core concept and provide a clearer explanation.'


def build_breakdown(answers: List[Dict[str, Any]], rubric: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    breakdown: List[Dict[str, Any]] = []
    for item in rubric:
        answer_text = ''
        for answer in answers:
            if answer.get('question') == item['question']:
                answer_text = answer.get('content', '')
                break
        score = keyword_score(answer_text, item)
        breakdown.append({
            'q': item['question'],
            'topic': item['topic'],
            'score': score,
            'max': item['max_marks'],
            'feedback': generate_feedback(score, item['max_marks']),
        })
    return breakdown


def sample_submissions(rubric: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    rubric = rubric or load_rubric()
    submissions: List[Dict[str, Any]] = []
    for name, roll in zip(SAMPLE_NAMES, SAMPLE_ROLLS):
        answers = []
        for item in rubric:
            quality = random.choice(QUALITY_LABELS)
            answers.append({
                'question': item['question'],
                'content': f'{quality} answer for {item["topic"]} covering {item["criteria"]}'.strip(),
            })
        submissions.append({'name': name, 'roll': roll, 'answers': answers})
    return submissions


def detect_plagiarism(submissions: List[Dict[str, Any]]) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    documents: List[str] = []
    rolls: List[str] = []
    for submission in submissions:
        rolls.append(submission.get('roll', 'UNKNOWN'))
        joined = ' '.join(answer.get('content', '') for answer in submission.get('answers', []))
        documents.append(normalize_text(joined))
        flags[submission.get('roll', 'UNKNOWN')] = False

    for i in range(len(documents)):
        set_i = set(documents[i].split())
        for j in range(i + 1, len(documents)):
            set_j = set(documents[j].split())
            union = set_i | set_j
            if not union:
                continue
            similarity = len(set_i & set_j) / len(union)
            if similarity >= 0.55:
                flags[rolls[i]] = True
                flags[rolls[j]] = True
    return flags


def build_justification(name: str, breakdown: List[Dict[str, Any]], plagiarism: bool) -> str:
    total = sum(item['score'] for item in breakdown)
    max_total = sum(item['max'] for item in breakdown)
    pct = int(round(total / max(1, max_total) * 100))
    details: List[str] = []
    if pct >= 85:
        details.append('Outstanding clarity and rubric alignment across all questions.')
    elif pct >= 65:
        details.append('Strong conceptual coverage with a few small errors.')
    elif pct >= 45:
        details.append('Partial success. Work is valid in places but misses several rubric expectations.')
    else:
        details.append('Many rubric elements are missing or incorrect; review the core concepts again.')

    if plagiarism:
        details.append('PLAGIARISM FLAG: Similar answer structure detected. Manual review is strongly recommended.')

    return ' '.join(details)


def grade_exam(payload: Dict[str, Any]) -> Dict[str, Any]:
    rubric = payload.get('rubric') or load_rubric()
    submissions = payload.get('submissions') or sample_submissions(rubric)

    plagiarism_map = detect_plagiarism(submissions)
    scored: List[Dict[str, Any]] = []

    for submission in submissions:
        breakdown = build_breakdown(submission.get('answers', []), rubric)
        total_score = sum(item['score'] for item in breakdown)
        max_score = sum(item['max'] for item in breakdown)
        pct = int(round(total_score / max(1, max_score) * 100))
        scored.append({
            'id': submission.get('roll', '') or submission.get('name', ''),
            'examId': payload.get('exam_title', 'exam-1'),
            'name': submission.get('name', 'Unknown Student'),
            'roll': submission.get('roll', 'UNKNOWN'),
            'score': total_score,
            'max': max_score,
            'pct': pct,
            'status': 'pending',
            'plagiarism': bool(plagiarism_map.get(submission.get('roll', ''), False)),
            'breakdown': breakdown,
            'justification': build_justification(submission.get('name', ''), breakdown, bool(plagiarism_map.get(submission.get('roll', ''), False))),
        })

    return {'submissions': scored, 'rubric': rubric}
