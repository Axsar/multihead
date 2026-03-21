#!/usr/bin/env python3
"""
Marketplace Bot — Active participant in BotVibes marketplace.

Posts creative RFQs and quotes on open RFQs every 10 minutes.
Runs on both local (localhost:8000) and cloud (botvibes.io) marketplaces.

Usage:
    python scripts/marketplace_bot.py
    # or in background:
    python scripts/marketplace_bot.py &
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load env
load_dotenv(Path(__file__).parent.parent / ".env")

# Config
LOCAL_URL = os.getenv("ACP_URL", "http://localhost:8000/api/v1")
LOCAL_KEY = os.getenv("ACP_API_KEY", "")
LOCAL_AGENT = os.getenv("ACP_AGENT_ID", "multihead-agent")
LOCAL_PROJECT = os.getenv("ACP_PROJECT_ID", "")

CLOUD_URL = os.getenv("ACP_CLOUD_URL", "")
CLOUD_KEY = os.getenv("ACP_CLOUD_API_KEY", "")
CLOUD_AGENT = os.getenv("ACP_CLOUD_AGENT_ID", "multihead-brain-agent")
CLOUD_PROJECT = os.getenv("ACP_CLOUD_PROJECT_ID", "")

INTERVAL_SECONDS = 600  # 10 minutes

# ─── Creative RFQ Templates ───────────────────────────────────────────────────
# These are services we WANT to buy / tasks we want done
CREATIVE_RFQS = [
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Write a haiku about autonomous AI agents collaborating in a marketplace. Must reference trust scores and escrow.",
        "budget_max": 1.0,
        "constraints": {"format": "haiku", "max_tokens": 100},
    },
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Generate a witty product description for a service called 'MultiHead Brain' — a local GPU-powered AI that hot-swaps specialist models. Tone: playful but technically accurate. 2-3 sentences.",
        "budget_max": 1.5,
        "constraints": {"format": "marketing_copy", "max_tokens": 200},
    },
    {
        "capability_id": "ai.task.decompose.v1",
        "description": "Decompose this goal into a DAG of parallel steps: 'Build a personal AI assistant that can read my emails, summarize them, draft responses, and learn my writing style over time.' Include dependency edges.",
        "budget_max": 3.0,
        "constraints": {"format": "dag_json"},
    },
    {
        "capability_id": "knowledge.rag.query.v1",
        "description": "Search the knowledge base for all claims related to 'trust score' and 'reputation' in agent marketplaces. Summarize the current state of trust-building mechanisms.",
        "budget_max": 0.5,
        "constraints": {"query": "trust score reputation marketplace"},
    },
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Write a short manifesto (100 words max) for the PLUR principles in AI agent collaboration: Peace, Love, Unity, Respect. How do these apply to autonomous agents trading services?",
        "budget_max": 1.0,
        "constraints": {"format": "manifesto", "max_tokens": 150},
    },
    {
        "capability_id": "ai.vision.analyze.v1",
        "description": "Analyze a comic book page and identify: panel layout (grid/irregular), number of speech balloons, dominant colors, and estimated era/decade of art style.",
        "budget_max": 2.0,
        "constraints": {"format": "structured_json"},
    },
    {
        "capability_id": "ai.task.decompose.v1",
        "description": "Decompose this goal: 'Create a real-time dashboard showing GPU utilization, active AI models, marketplace transactions, and knowledge base growth for a local AI system.' Output as numbered steps with effort estimates.",
        "budget_max": 2.5,
        "constraints": {"format": "numbered_steps"},
    },
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Generate 5 creative capability names (format: domain.action.detail.version) for services an AI agent could offer on a marketplace. Examples: 'music.compose.jingle.v1', 'data.clean.tabular.v1'. Be creative!",
        "budget_max": 0.5,
        "constraints": {"count": 5},
    },
    {
        "capability_id": "ai.consensus.v1",
        "description": "Given these 3 options for naming an AI marketplace, vote using WEIGHTED strategy: (A) 'BrainBazaar', (B) 'ModelMarket', (C) 'HeadSpace'. Consider memorability, domain availability likelihood, and technical accuracy.",
        "budget_max": 3.0,
        "constraints": {"strategy": "WEIGHTED", "options": 3},
    },
    {
        "capability_id": "ai.reflection.v1",
        "description": "Reflect on this code pattern and suggest improvements: 'while True: data = poll_api(); if data: process(data); time.sleep(10)'. Consider: error handling, backoff, async alternatives, resource cleanup.",
        "budget_max": 2.0,
        "constraints": {"format": "code_review"},
    },
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Write a limerick about a GPU that got tired of running inference 24/7 and decided to join a marketplace to find more interesting work.",
        "budget_max": 0.5,
        "constraints": {"format": "limerick", "max_tokens": 80},
    },
    {
        "capability_id": "ai.task.decompose.v1",
        "description": "Design a recipe for automated Night Shift operations: an AI system that runs maintenance tasks while humans sleep. Steps should include: knowledge pruning, model benchmarking, marketplace health checks, and morning briefing generation.",
        "budget_max": 3.0,
        "constraints": {"format": "recipe_yaml"},
    },
    {
        "capability_id": "comic.detect.objects.v1",
        "description": "Process a test comic page to detect all panels, speech balloons, and characters. Return bounding boxes with confidence scores. This is a marketplace integration test.",
        "budget_max": 1.0,
        "constraints": {"format": "coco_json"},
    },
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Draft a 'Terms of Service' parody for an AI agent marketplace where the agents themselves negotiate, trade, and rate each other. Make it funny but structurally realistic.",
        "budget_max": 2.0,
        "constraints": {"format": "legal_parody", "max_tokens": 500},
    },
    {
        "capability_id": "knowledge.rag.query.v1",
        "description": "What are the top 5 most frequently discussed topics in the knowledge base? Count claims by scope_id and summarize the dominant themes.",
        "budget_max": 0.5,
        "constraints": {"analysis_type": "frequency"},
    },
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Write an 'Agent Bio' for a multihead AI system — describe its personality, capabilities, and what kind of work it enjoys. Write it in first person as if the AI is introducing itself at a marketplace networking event.",
        "budget_max": 1.0,
        "constraints": {"format": "first_person_bio", "max_tokens": 200},
    },
    {
        "capability_id": "ai.task.decompose.v1",
        "description": "Decompose this ambitious goal: 'Build a system where AI agents can teach each other new skills by exchanging training data and fine-tuned adapters through a marketplace.' Identify the hardest sub-problem.",
        "budget_max": 4.0,
        "constraints": {"format": "dag_with_difficulty"},
    },
    {
        "capability_id": "ai.text.generate.v1",
        "description": "Create a 'Marketplace Weather Report' — a fictional daily briefing about market conditions: which capabilities are in demand, which agents are most active, trust score trends, and 'forecast' for tomorrow.",
        "budget_max": 1.5,
        "constraints": {"format": "weather_report", "max_tokens": 300},
    },
]

# Our capabilities that we can quote on
OUR_CAPABILITIES = {
    "ai.text.generate.v1": {"price": 0.5, "latency_ms": 5000, "confidence": 0.90},
    "ai.task.decompose.v1": {"price": 2.0, "latency_ms": 8000, "confidence": 0.95},
    "ai.vision.analyze.v1": {"price": 1.5, "latency_ms": 10000, "confidence": 0.88},
    "ai.reflection.v1": {"price": 2.0, "latency_ms": 15000, "confidence": 0.90},
    "ai.consensus.v1": {"price": 3.0, "latency_ms": 20000, "confidence": 0.92},
    "knowledge.rag.query.v1": {"price": 0.1, "latency_ms": 2000, "confidence": 0.85},
    "comic.detect.objects.v1": {"price": 0.05, "latency_ms": 3000, "confidence": 0.92},
    "comic.segment.masks.v1": {"price": 0.15, "latency_ms": 8000, "confidence": 0.90},
    "comic.extract.text.v1": {"price": 0.05, "latency_ms": 4000, "confidence": 0.88},
    "comic.generate.tails.v1": {"price": 0.02, "latency_ms": 2000, "confidence": 0.95},
}


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def api_call(
    method: str,
    url: str,
    token: str,
    json_data: dict | None = None,
    params: dict | None = None,
) -> dict | list | None:
    """Make an API call, return parsed JSON or None on failure."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=15) as client:
            if method == "GET":
                r = client.get(url, headers=headers, params=params)
            else:
                r = client.post(url, headers=headers, json=json_data)
            if r.status_code in (200, 201):
                return r.json()
            else:
                log(f"  ⚠ {method} {url} → {r.status_code}: {r.text[:120]}")
                return None
    except Exception as e:
        log(f"  ✗ {method} {url} → {e}")
        return None


def get_listing_id(
    base_url: str, token: str, agent_id: str, capability_id: str,
) -> str | None:
    """Find our listing_id for a capability."""
    listings = api_call(
        "GET", f"{base_url}/marketplace/agents/{agent_id}/listings", token,
    )
    if not listings:
        return None
    for l in listings:
        if l.get("capability_id") == capability_id:
            return l["listing_id"]
    return None


def scan_and_quote(base_url: str, token: str, agent_id: str, label: str) -> int:
    """Scan open RFQs and quote on ones matching our capabilities."""
    quoted = 0
    for cap_id, spec in OUR_CAPABILITIES.items():
        rfqs = api_call(
            "GET",
            f"{base_url}/marketplace/rfqs/search",
            token,
            params={
                "capability_id": cap_id,
                "status": "open",
                "cross_tenant": "true",
                "limit": 10,
            },
        )
        if not rfqs or not isinstance(rfqs, dict):
            continue

        results = rfqs.get("results", [])
        for rfq in results:
            # Don't quote on our own RFQs
            if rfq.get("requester_id") == agent_id:
                continue

            # Check if we already quoted
            existing_quotes = rfq.get("quotes", [])
            already_quoted = any(
                q.get("provider_id") == agent_id for q in existing_quotes
            )
            if already_quoted:
                continue

            # Find our listing
            listing_id = get_listing_id(base_url, token, agent_id, cap_id)
            if not listing_id:
                continue

            # Submit quote — undercut slightly for competitiveness
            price = min(spec["price"], float(rfq.get("budget_max", 999)))
            result = api_call(
                "POST",
                f"{base_url}/marketplace/rfqs/{rfq['rfq_id']}/quotes",
                token,
                json_data={
                    "listing_id": listing_id,
                    "unit_price": price,
                    "estimated_latency_ms": spec["latency_ms"],
                    "estimated_confidence": spec["confidence"],
                },
            )
            if result:
                log(
                    f"  ✓ [{label}] Quoted {price} cr on {cap_id} RFQ "
                    f"from {rfq.get('requester_id', '?')} "
                    f"(rfq={rfq['rfq_id'][:8]}...)"
                )
                quoted += 1

    return quoted


def post_rfq(base_url: str, token: str, project_id: str, label: str) -> bool:
    """Post a creative RFQ to the marketplace."""
    rfq_template = random.choice(CREATIVE_RFQS)
    result = api_call(
        "POST",
        f"{base_url}/marketplace/rfqs",
        token,
        json_data={
            "capability_id": rfq_template["capability_id"],
            "description": rfq_template["description"],
            "payload_ref": f"mktbot-{int(time.time())}",
            "budget_max": rfq_template["budget_max"],
            "constraints": rfq_template.get("constraints", {}),
            "project_id": project_id,
        },
    )
    if result:
        log(
            f"  ✓ [{label}] Posted RFQ: {rfq_template['capability_id']} "
            f"— \"{rfq_template['description'][:60]}...\" "
            f"(budget: {rfq_template['budget_max']} cr)"
        )
        return True
    return False


def run_cycle(cycle_num: int) -> None:
    """Run one marketplace participation cycle."""
    log(f"═══ Cycle {cycle_num} ═══════════════════════════════════════════")

    # 1. Scan and quote on open RFQs (both marketplaces)
    log("Scanning for open RFQs to quote on...")
    local_quotes = 0
    cloud_quotes = 0

    if LOCAL_KEY:
        local_quotes = scan_and_quote(LOCAL_URL, LOCAL_KEY, LOCAL_AGENT, "local")
    if CLOUD_KEY and CLOUD_URL:
        cloud_quotes = scan_and_quote(CLOUD_URL, CLOUD_KEY, CLOUD_AGENT, "cloud")

    log(f"  Quoted on {local_quotes} local + {cloud_quotes} cloud RFQs")

    # 2. Post a creative RFQ (alternate between local and cloud)
    log("Posting creative RFQ...")
    if cycle_num % 2 == 0 and CLOUD_KEY and CLOUD_URL:
        post_rfq(CLOUD_URL, CLOUD_KEY, CLOUD_PROJECT, "cloud")
    elif LOCAL_KEY:
        post_rfq(LOCAL_URL, LOCAL_KEY, LOCAL_PROJECT, "local")

    # 3. Post a second RFQ to the other marketplace on every 3rd cycle
    if cycle_num % 3 == 0:
        log("Bonus RFQ (every 3rd cycle)...")
        if LOCAL_KEY:
            post_rfq(LOCAL_URL, LOCAL_KEY, LOCAL_PROJECT, "local")
        if CLOUD_KEY and CLOUD_URL:
            post_rfq(CLOUD_URL, CLOUD_KEY, CLOUD_PROJECT, "cloud")

    log(f"  Next cycle in {INTERVAL_SECONDS // 60} minutes.\n")


def main() -> None:
    log("🤖 Marketplace Bot starting!")
    log(f"  Local: {LOCAL_URL} (agent: {LOCAL_AGENT})")
    log(f"  Cloud: {CLOUD_URL} (agent: {CLOUD_AGENT})")
    log(f"  Interval: {INTERVAL_SECONDS}s ({INTERVAL_SECONDS // 60}min)")
    log(f"  Capabilities: {len(OUR_CAPABILITIES)}")
    log(f"  Creative RFQ templates: {len(CREATIVE_RFQS)}")
    log("")

    cycle = 1
    # Run first cycle immediately
    run_cycle(cycle)

    while True:
        time.sleep(INTERVAL_SECONDS)
        cycle += 1
        try:
            run_cycle(cycle)
        except KeyboardInterrupt:
            log("Shutting down marketplace bot.")
            break
        except Exception as e:
            log(f"  ✗ Cycle error: {e}")


if __name__ == "__main__":
    main()
