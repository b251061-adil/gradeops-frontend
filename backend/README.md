# GradeOps ML Backend

This backend is a pure-Python grading service that simulates the GradeOps ML pipeline without external ML dependencies.

## Run the backend

```bash
cd backend
python main.py
```

Then open the frontend and use the Upload page. The frontend sends grading requests to `http://localhost:8000/grade`.

## API endpoints

- `GET /health` — returns basic health status
- `POST /grade` — grades the provided exam payload and returns submissions
- `POST /train` — no-op for this heuristic backend

## Payload structure

```json
{
  "exam_title": "Mid-Semester Examination",
  "total_marks": 50,
  "rubric": [ ... ],
  "submissions": [
    {
      "name": "Student A",
      "roll": "CS21B001",
      "answers": [
        { "question": "Q1", "content": "..." }
      ]
    }
  ]
}
```

If no submissions are provided, the backend generates sample submissions automatically.
