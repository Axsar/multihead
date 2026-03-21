"""Codebase scanner package — discover unique capabilities from project directories.

Re-exports all public API so callers can continue using:
    from multihead.codebase_scanner import CodebaseScanner, DiscoveredCapability, ScanResult
"""

from .models import (
    COMMODITY_PATTERNS,
    MAX_LINES,
    MAX_PY_FILES,
    MIN_FUNCTION_LINES,
    PRIORITY_DIRS,
    SKIP_DIRS,
    DiscoveredCapability,
    ScanResult,
)
from .parsers import (
    deduplicate,
    extract_all_metrics,
    infer_category,
    infer_model_type,
    metrics_near,
    parse_capability_doc,
    project_name,
    score_all,
    slug,
)
from .analyzers import (
    analyze_file,
    collect_python_files,
    find_models,
    model_to_capability,
    scan_python_files,
)
from .scanner import CodebaseScanner

__all__ = [
    # Models
    "DiscoveredCapability",
    "ScanResult",
    # Constants
    "COMMODITY_PATTERNS",
    "MAX_LINES",
    "MAX_PY_FILES",
    "MIN_FUNCTION_LINES",
    "PRIORITY_DIRS",
    "SKIP_DIRS",
    # Scanner
    "CodebaseScanner",
    # Analyzers
    "analyze_file",
    "collect_python_files",
    "find_models",
    "model_to_capability",
    "scan_python_files",
    # Parsers / helpers
    "deduplicate",
    "extract_all_metrics",
    "infer_category",
    "infer_model_type",
    "metrics_near",
    "parse_capability_doc",
    "project_name",
    "score_all",
    "slug",
]
