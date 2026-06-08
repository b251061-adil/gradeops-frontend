# GradeOps – Agentic Grading Pipeline

> **99% Automated Processing. 100% Human Accountability.**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATED MACHINE EXECUTION                  │
│                                                                 │
│  [1] Bulk Ingestion  →  [2] VLM Extraction  →  [3] Agentic Eval│
│   PDFs + JSON rubric    Nougat / Qwen-VL       LangChain +     │
│   uploaded by           transcribes messy      LangGraph award  │
│   instructor            handwriting            partial credit   │
│                                             + plagiarism detect │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                       HUMAN AUTHORITY                           │
│                                                                 │
│  [4] HITL Dashboard – TAs approve [ENTER] or override [SPACE]  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
gradeops/
├── schemas/
│   └── models.py              # Pydantic schemas (rubrics, results, grades)
├── pipelines/
│   ├── stage1_ingestion.py    # Bulk PDF + JSON rubric ingestion
│   ├── stage2_extraction.py   # VLM OCR (Nougat / Qwen-VL)
│   ├── stage3a_evaluation.py  # Agentic LLM grading (LangGraph)
│   └── stage3b_plagiarism.py  # Structural logic graph plagiarism detection
├── api/
│   └── hitl_dashboard.py      # FastAPI HITL review dashboard backend
├── orchestrator.py            # End-to-end pipeline runner + CLI
├── example_rubric.json        # Sample JSON rubric (MTH-101)
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your LLM API key
```bash
export OPENAI_API_KEY=sk-...
```

### 3. Start the HITL dashboard API
```bash
uvicorn api.hitl_dashboard:app --reload --port 8000
```

### 4. Run the pipeline via CLI
```bash
python orchestrator.py \
  --rubric example_rubric.json \
  --pdfs /path/to/exam/scans/ \
  --exam-id "MTH101-MIDTERM-2025" \
  --backend auto
```

### 5. Run programmatically
```python
from orchestrator import PipelineConfig, run_pipeline

config = PipelineConfig(
    rubric_json_path="example_rubric.json",
    exam_pdf_dir="/path/to/scans",
    exam_id="MTH101-MIDTERM-2025",
)
output = run_pipeline(config)
print(output.stats)
```

## Rubric Format

```json
{
  "rubric_id": "MTH-101-A",
  "course_id": "MTH101",
  "exam_name": "Midterm Exam",
  "total_points": 25.0,
  "criteria": [
    {
      "name": "Problem 1",
      "max_points": 10.0,
      "description": "...",
      "conditions": [
        {
          "condition_id": "P1_C1",
          "description": "Correct method selected",
          "points": 3.0,
          "is_partial_credit": false
        }
      ]
    }
  ]
}
```

## HITL Dashboard API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/results/load` | Load grading results into review queue |
| `GET`  | `/api/dashboard/queue` | Get pending submissions |
| `GET`  | `/api/dashboard/submission/{id}` | Get single submission for side-by-side review |
| `POST` | `/api/dashboard/review` | TA approve or override a grade |
| `GET`  | `/api/dashboard/stats` | Throughput & approval statistics |
| `GET`  | `/api/results/final` | Export finalized grades |

## VLM Backend Selection

| Backend | Best For | Model |
|---------|----------|-------|
| `nougat` | Printed/typed LaTeX-heavy documents | `facebook/nougat-base` |
| `qwen-vl` | Messy handwritten answers | `Qwen/Qwen-VL-Chat` |
| `auto` | Mixed content (default) | Heuristic selection |

## Plagiarism Detection

The system goes beyond text matching by mapping the **logical reasoning architecture** of every answer:

1. LLM extracts a directed logic graph from each answer (nodes = operations, edges = flow)
2. Pairwise Jaccard similarity is computed on operation multisets + edge patterns
3. Submissions exceeding the threshold (default 0.82) are clustered and flagged
4. Flags survive surface-level obfuscation (synonymous phrasing, different handwriting)
