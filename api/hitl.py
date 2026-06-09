"""
Vercel serverless handler for GradeOps HITL Dashboard API
Wraps the FastAPI app for Vercel's ASGI function support
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import gradeops modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl_dashboard import app

# Export the FastAPI app as the default handler for Vercel
# Vercel will automatically detect this ASGI app in the api/ directory
