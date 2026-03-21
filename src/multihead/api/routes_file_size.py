from fastapi import APIRouter
from ..file_size_scanner import scan_file_sizes

router = APIRouter()

@router.get("/scan")
async def scan():
    """Scan project directories for oversized files."""
    directories = [
        {"path": "src/multihead", "patterns": ["**/*.py"], "project": "multihead"},
    ]
    return {"findings": scan_file_sizes(directories)}
