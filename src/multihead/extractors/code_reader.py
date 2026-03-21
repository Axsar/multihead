"""Code reader channel — independent observation of source files.

Reads actual source code and produces claims about:
- Function signatures, parameters, return types
- Class hierarchies and relationships
- Import dependencies
- Module-level constants and configuration

SCOPING PRINCIPLE (independent channel):
This channel ONLY sees source code. It CANNOT see:
- Conversation transcripts
- Git history or commit messages
- Documentation or READMEs
- Other claims from the knowledge base

Independence is sacred — this channel must observe without bias
from other channels. Its claims are then fused with conversation
and git claims in the fusion step.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from multihead.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

# Scoped system prompt — strictly limited to code observation (AST/structural)
CODE_READER_PROMPT = """You are a code analysis system. Analyze ONLY the source code provided.

RULES:
- Report ONLY what is directly visible in the code
- Do NOT reference conversations, documentation, or external context
- Do NOT guess the purpose of code from function names alone — describe behavior from the code
- If something is ambiguous from the code alone, say UNRESOLVABLE, not your best guess
- Each claim must cite the specific file and line number

For each observation, output a JSON object:
{{"claim_type":"fact","claim_key":"<module>.<class_or_function>.<aspect>","statement":"<what the code shows>","confidence":0.85,"file_path":"<path>","symbol":"<function_or_class_name>","line":"<line_number>","observation_method":"code_read","speaker":"tool","evidence":[{{"type":"code_read","file":"<path>","line":"<N>","text":"<relevant code snippet>"}}]}}

Output ONLY a JSON array of observations. No commentary.

Source code:
{code}
"""

# Behavioral analysis prompt — focuses on WHAT code DOES, not just structure
CODE_BEHAVIOR_PROMPT = """You are a behavioral code analysis system. For each function and class in the source code below, describe what it ACTUALLY DOES — not just its signature.

RULES:
- Report ONLY what is directly observable in the code
- Do NOT reference conversations, documentation, or external context
- Focus on BEHAVIOR: what happens when this code runs?
- If something is ambiguous from the code alone, say UNRESOLVABLE

For each function/class, analyze:
1. What does this function/class actually do? Describe the BEHAVIOR, not just the signature.
2. What side effects does it have? (file I/O, network calls, state mutations, logging)
3. What invariants does it maintain? (preconditions, postconditions, internal consistency checks)
4. What error conditions does it handle? (try/except, early returns, validation)

For each observation, output a JSON object:
{{"claim_type":"fact","claim_key":"<module>.<symbol>.behavior","statement":"<behavioral description>","file_path":"{file_path}","symbol":"<function_or_class_name>","line":"<line_number>","evidence":[{{"type":"code_read","file":"{file_path}","line":"<N>","text":"<relevant code snippet>"}}]}}

Output ONLY a JSON array of observations. No commentary.

Source code from {file_path}:
```python
{code}
```
"""


def extract_claims_from_file(
    file_path: str,
    adapter=None,
    max_lines: int = 500,
) -> list[dict]:
    """Extract claims from a single source file.

    Two modes:
    1. AST-based (no LLM) — fast, extracts structural facts
    2. LLM-based (with adapter) — deeper analysis of behavior

    AST mode is always run. LLM mode runs if adapter is provided.
    """
    path = Path(file_path)
    if not path.exists() or not path.suffix == ".py":
        return []

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = source.split("\n")
    if len(lines) > max_lines:
        source = "\n".join(lines[:max_lines])

    # Always run AST extraction (free, no LLM)
    claims = _ast_extract(file_path, source)

    return claims


def _ast_extract(file_path: str, source: str) -> list[dict]:
    """Extract structural claims from Python source via AST.

    No LLM needed — pure static analysis. Produces claims about:
    - Function names, parameters, decorators
    - Class names, bases, methods
    - Imports
    - Module-level assignments
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    claims: list[dict] = []
    module_name = Path(file_path).stem

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Function signature
            params = []
            for arg in node.args.args:
                param_name = arg.arg
                annotation = ""
                if arg.annotation:
                    try:
                        annotation = ast.unparse(arg.annotation)
                    except Exception:
                        pass
                params.append(f"{param_name}: {annotation}" if annotation else param_name)

            # Return type
            returns = ""
            if node.returns:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:
                    pass

            param_str = ", ".join(params)
            sig = f"{node.name}({param_str})"
            if returns:
                sig += f" -> {returns}"

            # Decorators
            decorators = []
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except Exception:
                    pass

            statement = f"Function {sig} defined at line {node.lineno}"
            if decorators:
                statement += f" with decorators: {', '.join(decorators)}"

            # Get the first line of docstring if present
            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.split("\n")[0].strip()
                statement += f". Purpose: {first_line}"

            if len(statement) >= 50:
                claims.append({
                    "claim_type": "fact",
                    "claim_key": f"{module_name}.{node.name}.signature",
                    "statement": statement,
                    "confidence": 0.95,  # AST is ground truth
                    "file_path": file_path,
                    "symbol": node.name,
                    "line": str(node.lineno),
                    "observation_method": "code_read",
                    "speaker": "tool",
                    "evidence": [{
                        "type": "code_read",
                        "file": file_path,
                        "line": str(node.lineno),
                        "text": f"def {sig}",
                    }],
                    "durability": "durable",
                })

        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    pass

            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]

            statement = f"Class {node.name} defined at line {node.lineno}"
            if bases:
                statement += f", inherits from {', '.join(bases)}"
            if methods:
                statement += f". Methods: {', '.join(methods[:10])}"
                if len(methods) > 10:
                    statement += f" (+{len(methods)-10} more)"

            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.split("\n")[0].strip()
                statement += f". Purpose: {first_line}"

            if len(statement) >= 50:
                claims.append({
                    "claim_type": "definition",
                    "claim_key": f"{module_name}.{node.name}.definition",
                    "statement": statement,
                    "confidence": 0.95,
                    "file_path": file_path,
                    "symbol": node.name,
                    "line": str(node.lineno),
                    "observation_method": "code_read",
                    "speaker": "tool",
                    "evidence": [{
                        "type": "code_read",
                        "file": file_path,
                        "line": str(node.lineno),
                        "text": f"class {node.name}({', '.join(bases)})",
                    }],
                    "durability": "durable",
                })

    return claims


def scan_project(
    project_root: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_files: int = 100000,
) -> list[dict]:
    """Scan a project directory and extract code claims from Python files.

    This is the independent code observation channel.
    No conversation context, no git history — pure code analysis.
    """
    root = Path(project_root)
    exclude = set(exclude_patterns or [
        "__pycache__", ".git", ".venv", "venv", "env",
        "node_modules", ".tox", "site-packages",
        "build", "dist", "*.egg-info", ".eggs",
        "migrations", "vendor", "third_party",
    ])

    claims: list[dict] = []
    files_scanned = 0

    for py_file in sorted(root.rglob("*.py")):
        # Skip excluded dirs
        if any(excl in str(py_file) for excl in exclude):
            continue

        if files_scanned >= max_files:
            break

        file_claims = extract_claims_from_file(str(py_file))
        claims.extend(file_claims)
        files_scanned += 1

    if files_scanned > 10000:
        logger.warning("Code reader scanned large file set: %d files, %d claims", files_scanned, len(claims))
    logger.info("Code reader: scanned %d files, produced %d claims", files_scanned, len(claims))
    return claims


# ---------------------------------------------------------------------------
# Behavioral LLM channel — independent from AST extraction
# ---------------------------------------------------------------------------


async def extract_behavioral_claims(
    file_path: str,
    adapter,
    max_lines: int = 300,
) -> list[dict]:
    """Extract behavioral claims from a source file using an LLM.

    This is the behavioral observation channel — it analyzes WHAT code does,
    not just its structure.  Independent from AST extraction.

    Args:
        file_path: Path to a Python source file.
        adapter: A HeadAdapter or async generate function.
        max_lines: Truncate source beyond this many lines.

    Returns:
        List of claim dicts with observation_method="code_behavior_llm".
    """
    if adapter is None:
        return []

    path = Path(file_path)
    if not path.exists() or path.suffix != ".py":
        return []

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Cannot read %s for behavioral analysis", file_path)
        return []

    lines = source.split("\n")
    if len(lines) > max_lines:
        source = "\n".join(lines[:max_lines])

    # Skip tiny files — not worth an LLM call
    if len(source.strip()) < 50:
        return []

    module_name = Path(file_path).stem
    prompt = CODE_BEHAVIOR_PROMPT.format(file_path=file_path, code=source)

    try:
        result = await BaseExtractor.call_generate(adapter, prompt)
        raw_text = result.get("text", "") if isinstance(result, dict) else str(result)
    except Exception as exc:
        logger.warning("LLM behavioral extraction failed for %s: %s", file_path, exc)
        return []

    parsed = BaseExtractor.parse_json_response(raw_text)
    if not parsed:
        logger.debug("No behavioral claims parsed from LLM response for %s", file_path)
        return []

    # Normalize each claim with behavioral channel metadata
    claims: list[dict] = []
    for item in parsed:
        symbol = item.get("symbol", "unknown")
        # Ensure claim_key follows the behavioral pattern
        claim_key = item.get("claim_key", "")
        if not claim_key or not claim_key.endswith(".behavior"):
            claim_key = f"{module_name}.{symbol}.behavior"

        claims.append({
            "claim_type": item.get("claim_type", "fact"),
            "claim_key": claim_key,
            "statement": item.get("statement", ""),
            "confidence": 0.75,  # LLM inference — lower than AST ground truth
            "file_path": file_path,
            "symbol": symbol,
            "line": str(item.get("line", "0")),
            "observation_method": "code_behavior_llm",
            "speaker": "tool",
            "evidence": item.get("evidence", [{
                "type": "code_read",
                "file": file_path,
                "line": str(item.get("line", "0")),
                "text": f"behavioral analysis of {symbol}",
            }]),
            "durability": "session",  # LLM claims may drift; not as durable as AST
        })

    logger.info(
        "Behavioral extraction: %d claims from %s", len(claims), file_path,
    )
    return claims


async def scan_project_behavioral(
    project_root: str,
    adapter,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_files: int = 100000,
    concurrency: int = 3,
    batch_mode: bool = False,
    no_wait: bool = False,
) -> list[dict]:
    """Scan a project and extract behavioral claims using an LLM.

    Like scan_project() but uses the behavioral LLM channel instead of AST.
    Processes files concurrently via BaseExtractor.map_generate().

    Args:
        project_root: Root directory to scan.
        adapter: LLM adapter (HeadAdapter or async callable). If None, returns [].
        include_patterns: Glob patterns to include (not yet used, reserved).
        exclude_patterns: Directory names to skip.
        max_files: Maximum number of files to process.
        concurrency: How many LLM calls to run in parallel.

    Returns:
        List of behavioral claim dicts.
    """
    if adapter is None:
        logger.info("Behavioral scan skipped — no adapter provided")
        return []

    root = Path(project_root)
    exclude = set(exclude_patterns or [
        "__pycache__", ".git", ".venv", "venv", "env",
        "node_modules", ".tox", "site-packages",
        "build", "dist", "*.egg-info", ".eggs",
        "migrations", "vendor", "third_party",
    ])

    # Collect files to analyze
    py_files: list[str] = []
    for py_file in sorted(root.rglob("*.py")):
        if any(excl in str(py_file) for excl in exclude):
            continue
        if len(py_files) >= max_files:
            break
        # Skip tiny files
        try:
            if py_file.stat().st_size < 50:
                continue
        except OSError:
            continue
        py_files.append(str(py_file))

    if not py_files:
        return []

    logger.info(
        "Behavioral scan: analyzing %d files with concurrency=%d",
        len(py_files), concurrency,
    )

    # Build prompts for map_generate
    prompts: list[str] = []
    file_sources: list[tuple[str, str]] = []  # (file_path, source) for post-processing
    for fp in py_files:
        try:
            source = Path(fp).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(source.strip()) < 50:
            continue
        prompts.append(CODE_BEHAVIOR_PROMPT.format(file_path=fp, code=source))
        file_sources.append((fp, source))

    if not prompts:
        return []

    # Use map_generate — batch mode for 50% cheaper, no_wait to submit and exit
    results = await BaseExtractor.map_generate(
        adapter, prompts, concurrency=concurrency,
        stage_name="behavioral_scan",
        batch_mode=batch_mode,
        no_wait=no_wait,
    )

    # Parse results and normalize claims
    all_claims: list[dict] = []
    for idx, result in enumerate(results):
        fp = file_sources[idx][0]
        module_name = Path(fp).stem

        if isinstance(result, Exception):
            logger.warning("Behavioral LLM call failed for %s: %s", fp, result)
            continue

        raw_text = result.get("text", "") if isinstance(result, dict) else str(result)
        parsed = BaseExtractor.parse_json_response(raw_text)

        for item in parsed:
            symbol = item.get("symbol", "unknown")
            claim_key = item.get("claim_key", "")
            if not claim_key or not claim_key.endswith(".behavior"):
                claim_key = f"{module_name}.{symbol}.behavior"

            all_claims.append({
                "claim_type": item.get("claim_type", "fact"),
                "claim_key": claim_key,
                "statement": item.get("statement", ""),
                "confidence": 0.75,
                "file_path": fp,
                "symbol": symbol,
                "line": str(item.get("line", "0")),
                "observation_method": "code_behavior_llm",
                "speaker": "tool",
                "evidence": item.get("evidence", [{
                    "type": "code_read",
                    "file": fp,
                    "line": str(item.get("line", "0")),
                    "text": f"behavioral analysis of {symbol}",
                }]),
                "durability": "session",
            })

    if len(file_sources) > 10000:
        logger.warning("Behavioral scan processed large file set: %d files, %d claims", len(file_sources), len(all_claims))
    logger.info(
        "Behavioral scan complete: %d claims from %d files",
        len(all_claims), len(file_sources),
    )
    return all_claims
