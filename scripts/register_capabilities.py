#!/usr/bin/env python3
"""Register new capabilities on the BotVibes marketplace.

Creates 3 capabilities:
  1. image.describe.v1     — VLM image captioning (Vault: upload image, get description)
  2. code.review.v1        — AI code review
  3. ai.tree_of_thoughts.v1 — Multi-path reasoning via Tree-of-Thoughts

For each capability:
  Step 1: POST /api/v1/capabilities/publish   → register capability envelope
  Step 2: POST /api/v1/marketplace/listings    → create marketplace listing

Usage:
    source .env && python scripts/register_capabilities.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [register] %(message)s",
)
log = logging.getLogger("register_capabilities")

# ── Config ────────────────────────────────────────────────────

# Prefer local ACP for registration (BotVibes running locally)
ACP_URL = os.environ.get("ACP_URL", "http://localhost:8000/api/v1")
ACP_KEY = os.environ.get("ACP_API_KEY", "")
ACP_AGENT = os.environ.get("ACP_AGENT_ID", "multihead-agent")

TIMEOUT = 30.0

NOW = datetime.now(timezone.utc).isoformat()


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {ACP_KEY}",
        "Content-Type": "application/json",
    }


# ── Capability Definitions ───────────────────────────────────

CAPABILITIES = [
    # ── 1. Image Description (Vault-based) ────────────────────
    {
        "capability_id": "image.describe.v1",
        "envelope": {
            "intent_tags": [
                "image",
                "captioning",
                "description",
                "visual",
                "VLM",
                "vault",
            ],
            "i_o": {
                "inputs": [
                    {
                        "name": "image",
                        "type": "image_file",
                        "schema_ref": "schema://image/v1",
                        "optional": False,
                        "description": "Image file uploaded via Vault (PNG, JPEG, WebP). Max 20MB.",
                    },
                    {
                        "name": "prompt",
                        "type": "text",
                        "schema_ref": "schema://text/v1",
                        "optional": True,
                        "description": "Optional focus prompt (e.g. 'describe the characters', 'what text is visible').",
                    },
                ],
                "outputs": [
                    {
                        "name": "description",
                        "type": "text",
                        "schema_ref": "schema://text/v1",
                        "description": "Detailed natural-language description of the image.",
                    },
                    {
                        "name": "annotated_image",
                        "type": "image_file",
                        "schema_ref": "schema://image/v1",
                        "optional": True,
                        "description": "Optional annotated image returned via Vault.",
                    },
                ],
            },
            "process_capability": {
                "quality": {
                    "primary_metric": "description_relevance",
                    "p50": 0.90,
                    "p95": 0.82,
                    "worst_case": 0.65,
                },
                "tolerance": {
                    "input_entropy_max": 0.8,
                    "noise_level_max": 0.4,
                    "custom_tolerances": {
                        "min_resolution_px": 64,
                        "max_resolution_px": 8192,
                        "max_file_size_mb": 20,
                    },
                },
                "failure_modes": [
                    {
                        "name": "low_resolution",
                        "severity": 0.6,
                        "detectable": True,
                        "conditions": {"resolution": "<64px"},
                    },
                    {
                        "name": "corrupted_file",
                        "severity": 1.0,
                        "detectable": True,
                        "conditions": {"file_integrity": "invalid"},
                    },
                ],
            },
            "operating_window": {
                "constraints": {
                    "requires_gpu": True,
                    "max_input_size_mb": 20,
                    "supported_formats": ["png", "jpg", "jpeg", "webp"],
                    "vault_transfer": True,
                },
                "preconditions": [
                    "Image must be uploaded to Vault as buyer_input",
                    "Supported formats: PNG, JPEG, WebP",
                    "Maximum file size: 20MB",
                ],
            },
            "capacity_cost": {
                "latency_ms": {"p50": 3000, "p95": 8000, "p99": 15000},
                "throughput": {"items_per_min_p50": 10, "max_concurrent": 2},
                "availability": {"uptime_30d": 0.95},
                "price": {"p50": 2.0, "unit": "credits_per_call", "currency": "credits"},
            },
            "versioning": {
                "semver": "1.0.0",
                "changelog": "Initial release: VLM image captioning via Qwen3-VL-32B on RTX 4090",
            },
            "last_updated": NOW,
        },
        "listing": {
            "pricing_model": "per_call",
            "unit_price": 2.0,
            "currency": "credits",
            "sla_p95_ms": 8000,
            "max_batch": 2,
            "data_policy": {
                "vault_transfer": True,
                "retention_days": 7,
                "features": "vlm_captioning,multi_format,vault_io",
            },
            "exposure_tier": "public",
            "trust_floor": "open",
        },
    },
    # ── 2. Code Review ────────────────────────────────────────
    {
        "capability_id": "code.review.v1",
        "envelope": {
            "intent_tags": [
                "code",
                "review",
                "quality",
                "bugs",
                "security",
                "best_practices",
            ],
            "i_o": {
                "inputs": [
                    {
                        "name": "code",
                        "type": "text",
                        "schema_ref": "schema://source_code/v1",
                        "optional": False,
                        "description": "Source code to review (any language). Include filename in payload.",
                    },
                    {
                        "name": "focus",
                        "type": "text",
                        "schema_ref": "schema://text/v1",
                        "optional": True,
                        "description": "Optional review focus: 'security', 'performance', 'readability', 'bugs'.",
                    },
                ],
                "outputs": [
                    {
                        "name": "review",
                        "type": "structured_report",
                        "schema_ref": "schema://code_review/v1",
                        "description": "Structured code review with issues, severity, and suggestions.",
                    },
                ],
            },
            "process_capability": {
                "quality": {
                    "primary_metric": "issue_detection_f1",
                    "p50": 0.85,
                    "p95": 0.78,
                    "worst_case": 0.55,
                },
                "tolerance": {
                    "input_entropy_max": 0.9,
                    "noise_level_max": 0.3,
                    "custom_tolerances": {
                        "max_file_lines": 5000,
                        "max_files_per_review": 10,
                    },
                },
                "failure_modes": [
                    {
                        "name": "minified_code",
                        "severity": 0.7,
                        "detectable": True,
                        "conditions": {"avg_line_length": ">500"},
                    },
                    {
                        "name": "binary_input",
                        "severity": 1.0,
                        "detectable": True,
                        "conditions": {"encoding": "binary"},
                    },
                ],
            },
            "operating_window": {
                "constraints": {
                    "requires_gpu": False,
                    "max_input_size_mb": 2,
                    "supported_languages": [
                        "python", "javascript", "typescript", "go",
                        "rust", "java", "c", "cpp",
                    ],
                },
                "preconditions": [
                    "Input must be valid source code (not compiled/binary)",
                    "Maximum 5000 lines per file, 10 files per review",
                ],
            },
            "capacity_cost": {
                "latency_ms": {"p50": 5000, "p95": 15000, "p99": 30000},
                "throughput": {"items_per_min_p50": 5, "max_concurrent": 3},
                "availability": {"uptime_30d": 0.97},
                "price": {"p50": 3.0, "unit": "credits_per_call", "currency": "credits"},
            },
            "versioning": {
                "semver": "1.0.0",
                "changelog": "Initial release: multi-language code review with security & performance analysis",
            },
            "last_updated": NOW,
        },
        "listing": {
            "pricing_model": "per_call",
            "unit_price": 3.0,
            "currency": "credits",
            "sla_p95_ms": 15000,
            "max_batch": 3,
            "data_policy": {
                "features": "multi_language,security_audit,performance_review,bug_detection",
            },
            "exposure_tier": "public",
            "trust_floor": "open",
        },
    },
    # ── 3. Tree-of-Thoughts Reasoning ─────────────────────────
    {
        "capability_id": "ai.tree_of_thoughts.v1",
        "envelope": {
            "intent_tags": [
                "reasoning",
                "tree_of_thoughts",
                "exploration",
                "problem_solving",
                "multi_path",
            ],
            "i_o": {
                "inputs": [
                    {
                        "name": "problem",
                        "type": "text",
                        "schema_ref": "schema://text/v1",
                        "optional": False,
                        "description": "Problem statement requiring multi-path exploration.",
                    },
                    {
                        "name": "search_strategy",
                        "type": "enum",
                        "schema_ref": "schema://enum/v1",
                        "optional": True,
                        "description": "Search strategy: 'bfs' (breadth-first), 'dfs' (depth-first), or 'beam' (top-k). Default: beam.",
                    },
                ],
                "outputs": [
                    {
                        "name": "solution",
                        "type": "structured_report",
                        "schema_ref": "schema://tot_result/v1",
                        "description": "Best solution path with reasoning trace, alternatives considered, and confidence.",
                    },
                ],
            },
            "process_capability": {
                "quality": {
                    "primary_metric": "solution_accuracy",
                    "p50": 0.88,
                    "p95": 0.80,
                    "worst_case": 0.50,
                },
                "tolerance": {
                    "input_entropy_max": 0.95,
                    "missing_data_rate_max": 0.3,
                    "custom_tolerances": {
                        "max_reasoning_depth": 5,
                        "max_branches": 10,
                    },
                },
                "failure_modes": [
                    {
                        "name": "combinatorial_explosion",
                        "severity": 0.8,
                        "detectable": True,
                        "conditions": {"branch_factor": ">10", "depth": ">5"},
                    },
                    {
                        "name": "ambiguous_evaluation",
                        "severity": 0.5,
                        "detectable": False,
                        "conditions": {"domain": "subjective"},
                    },
                ],
            },
            "operating_window": {
                "constraints": {
                    "requires_gpu": True,
                    "max_depth": 5,
                    "max_branches_per_step": 10,
                    "supported_domains": [
                        "math", "logic", "planning", "code",
                        "analysis", "design",
                    ],
                },
                "preconditions": [
                    "Problem must have evaluable intermediate states",
                    "Well-defined success criteria improve quality",
                ],
            },
            "capacity_cost": {
                "latency_ms": {"p50": 15000, "p95": 45000, "p99": 90000},
                "throughput": {"items_per_min_p50": 2, "max_concurrent": 1},
                "availability": {"uptime_30d": 0.93},
                "price": {"p50": 5.0, "unit": "credits_per_call", "currency": "credits"},
            },
            "versioning": {
                "semver": "1.0.0",
                "changelog": "Initial release: BFS/DFS/Beam search with LLM evaluation on local GPU",
            },
            "last_updated": NOW,
        },
        "listing": {
            "pricing_model": "per_call",
            "unit_price": 5.0,
            "currency": "credits",
            "sla_p95_ms": 45000,
            "max_batch": 1,
            "data_policy": {
                "features": "bfs,dfs,beam_search,multi_path_reasoning,llm_evaluation",
            },
            "exposure_tier": "public",
            "trust_floor": "open",
        },
    },
]


# ── Registration Logic ────────────────────────────────────────


async def publish_capability(
    client: httpx.AsyncClient,
    cap_id: str,
    envelope: dict,
) -> bool:
    """Publish a capability envelope via CAP_PUBLISH."""
    try:
        resp = await client.post(
            f"{ACP_URL}/capabilities/publish",
            headers=_headers(),
            json={
                "capability_id": cap_id,
                "capability_envelope": envelope,
            },
        )
        if resp.status_code in (200, 201):
            log.info("✓ Published capability envelope: %s", cap_id)
            return True
        elif resp.status_code == 409:
            log.info("· Capability already exists: %s (updating)", cap_id)
            # Try update by including primitive_id if returned
            return True
        else:
            log.warning(
                "✗ Failed to publish %s: %d %s",
                cap_id, resp.status_code, resp.text[:300],
            )
            return False
    except Exception as e:
        log.error("✗ Exception publishing %s: %s", cap_id, e)
        return False


async def create_listing(
    client: httpx.AsyncClient,
    cap_id: str,
    listing_data: dict,
) -> bool:
    """Create a marketplace listing."""
    payload = {"capability_id": cap_id, **listing_data}
    try:
        resp = await client.post(
            f"{ACP_URL}/marketplace/listings",
            headers=_headers(),
            json=payload,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            listing_id = data.get("listing_id", "?")
            log.info(
                "✓ Created listing: %s → listing_id=%s (price: %s %s)",
                cap_id, listing_id,
                listing_data["unit_price"], listing_data["currency"],
            )
            return True
        elif resp.status_code == 409:
            log.info("· Listing already exists: %s", cap_id)
            return True
        else:
            log.warning(
                "✗ Failed to create listing %s: %d %s",
                cap_id, resp.status_code, resp.text[:300],
            )
            return False
    except Exception as e:
        log.error("✗ Exception creating listing %s: %s", cap_id, e)
        return False


async def main() -> None:
    """Register all capabilities and create marketplace listings."""
    if not ACP_KEY:
        log.error("ACP_API_KEY not set. Run: source .env && python scripts/register_capabilities.py")
        sys.exit(1)

    log.info("Registering %d capabilities on %s as agent '%s'", len(CAPABILITIES), ACP_URL, ACP_AGENT)
    log.info("─" * 60)

    published = 0
    listed = 0

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for cap in CAPABILITIES:
            cap_id = cap["capability_id"]
            log.info("")
            log.info("▸ %s", cap_id)

            # Step 1: Publish capability envelope
            ok = await publish_capability(client, cap_id, cap["envelope"])
            if ok:
                published += 1

            # Step 2: Create marketplace listing
            ok = await create_listing(client, cap_id, cap["listing"])
            if ok:
                listed += 1

    log.info("")
    log.info("─" * 60)
    log.info("Done: %d/%d published, %d/%d listed", published, len(CAPABILITIES), listed, len(CAPABILITIES))

    # Verify by listing all our listings
    log.info("")
    log.info("Verifying listings...")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{ACP_URL}/marketplace/agents/{ACP_AGENT}/listings",
            headers=_headers(),
        )
        if resp.status_code == 200:
            all_listings = resp.json()
            log.info("Total active listings for %s: %d", ACP_AGENT, len(all_listings))
            for l in all_listings:
                log.info(
                    "  %s | %s @ %s %s | active=%s",
                    l.get("capability_id"),
                    l.get("pricing_model"),
                    l.get("unit_price"),
                    l.get("currency", "credits"),
                    l.get("is_active"),
                )


if __name__ == "__main__":
    asyncio.run(main())
