import os
from typing import Optional, List
from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import Column
from sqlalchemy.types import JSON

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gradeops")
# Clean up asyncpg schema if present, since we will use synchronous engine for FastAPI sync endpoints
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

# SQLite fallback support for local development/testing
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

class GradingResultRow(SQLModel, table=True):
    __tablename__ = "grading_result"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    submission_id: str = Field(index=True, unique=True)
    student_id: str
    rubric_id: str
    exam_id: Optional[str] = None
    total_score: float
    max_score: float
    criterion_results: List[dict] = Field(sa_column=Column(JSON))
    plagiarism_flag: bool = False
    plagiarism_similar_to: List[str] = Field(sa_column=Column(JSON))
    evaluation_model: str
    status: str = "proposed"

class FinalGradeRow(SQLModel, table=True):
    __tablename__ = "final_grade"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    submission_id: str = Field(index=True, unique=True)
    student_id: str
    final_score: float
    max_score: float
    reviewed_by: str
    action_taken: str
    ai_proposed_score: float
