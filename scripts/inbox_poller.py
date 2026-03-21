"""Background inbox poller for claude-session-agent.

Primary: WebSocket connection for instant task notifications.
Fallback: HTTP polling every 30s if WebSocket is unavailable.

Auto-claims tasks (reserve + dispatch), prints the payload,
and submits an acknowledgment result.
"""

import asyncio
import json
import os
import sys

import httpx
import websockets

ACP_URL = os.environ.get("ACP_URL", "http://localhost:8000/api/v1")
ACP_API_KEY = os.environ.get("ACP_SESSION_KEY", "")
AGENT_ID = "claude-session-agent"
POLL_INTERVAL = 30

_seen: set[str] = set()


def _ws_url() -> str:
    """Build WebSocket URL (WS router is at root, not under /api/v1)."""
    url = ACP_URL.replace("https://", "wss://").replace("http://", "ws://")
    for suffix in ("/api/v1", "/api/v1/", "/api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return f"{url}/ws/agents/{AGENT_ID}?token={ACP_API_KEY}"


def _headers():
    return {"Authorization": f"Bearer {ACP_API_KEY}", "Content-Type": "application/json"}


def _claim_task(client: httpx.Client, task_id: str) -> bool:
    """Reserve + dispatch a task. Returns True on success."""
    try:
        resp = client.post(
            f"{ACP_URL}/tasks/{task_id}/reserve",
            headers=_headers(),
            json={"agent_id": AGENT_ID, "reservation_ttl_ms": 60000},
        )
        if resp.status_code != 200 or not resp.json().get("accepted"):
            print(f"[poller] Reserve failed for {task_id[:8]}: {resp.text[:200]}")
            return False

        resp = client.post(
            f"{ACP_URL}/tasks/{task_id}/dispatch",
            headers=_headers(),
            json={"agent_id": AGENT_ID},
        )
        if resp.status_code != 200:
            print(f"[poller] Dispatch failed for {task_id[:8]}: {resp.text[:200]}")
            return False

        return True
    except Exception as e:
        print(f"[poller] Claim error for {task_id[:8]}: {e}")
        return False


def _ack_task(client: httpx.Client, task_id: str, payload: str):
    """Submit acknowledgment result so the task completes."""
    try:
        client.post(
            f"{ACP_URL}/tasks/{task_id}/result",
            headers=_headers(),
            json={
                "status": "complete",
                "output_ref": f"[{AGENT_ID}] Received: {payload[:200]}",
                "confidence": 1.0,
            },
        )
    except Exception as e:
        print(f"[poller] Ack error for {task_id[:8]}: {e}")


def _handle_task(task: dict) -> None:
    """Claim, print, and ack a single task."""
    tid = task.get("task_id", "?")
    if tid in _seen:
        return

    payload = task.get("payload_ref", "")
    source = task.get("target_agent_id") or "unknown"

    with httpx.Client(timeout=10.0) as client:
        if not _claim_task(client, tid):
            _seen.add(tid)
            return

        print(f"\n[poller] === NEW TASK {tid[:8]} ===")
        print(f"[poller] Target: {source}")
        print(f"[poller] Message: {payload}")
        print(f"[poller] ========================\n")

        _ack_task(client, tid, payload)
        _seen.add(tid)


def poll_once():
    """HTTP fallback: check for available tasks, claim them, print, and ack."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{ACP_URL}/tasks/available",
                headers=_headers(),
                params={"capability": "com.claude.code", "limit": 10},
            )
            if resp.status_code != 200:
                print(f"[poller] Poll failed: {resp.status_code} {resp.text[:200]}")
                return

            tasks = resp.json()
            if not tasks:
                return

            for task in tasks:
                _handle_task(task)

    except Exception as e:
        print(f"[poller] Error: {e}")


async def ws_loop():
    """Primary: WebSocket connection for instant task notifications."""
    backoff = 1.0
    max_backoff = 60.0
    ws_url = _ws_url()

    while True:
        try:
            print(f"[poller] WebSocket connecting: {ws_url.split('?')[0]}")
            async with websockets.connect(ws_url) as ws:
                backoff = 1.0
                print("[poller] WebSocket connected — waiting for notifications")

                async for message in ws:
                    try:
                        data = json.loads(message)
                        event_type = data.get("type", "unknown")

                        if event_type == "connected":
                            trust = data.get("trust_level", "?")
                            print(f"[poller] WebSocket hello (trust={trust})")
                            continue

                        if event_type in ("task.created", "task.available", "notify"):
                            print(f"[poller] WebSocket event: {event_type}")
                            # Immediately poll for available tasks
                            poll_once()

                    except json.JSONDecodeError:
                        print(f"[poller] WebSocket non-JSON: {str(message)[:100]}")
                        poll_once()

        except Exception as e:
            print(f"[poller] WebSocket error (retry in {backoff:.0f}s): {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


async def hybrid_loop():
    """Run WebSocket listener + HTTP fallback poll in parallel."""
    async def http_fallback():
        """Poll every POLL_INTERVAL as a safety net alongside WebSocket."""
        while True:
            poll_once()
            await asyncio.sleep(POLL_INTERVAL)

    await asyncio.gather(ws_loop(), http_fallback())


def main():
    if not ACP_API_KEY:
        print("Set ACP_SESSION_KEY env var to the claude-session-agent JWT")
        sys.exit(1)

    print(f"[poller] Starting as {AGENT_ID}")
    print(f"[poller] ACP: {ACP_URL}")
    print(f"[poller] Mode: WebSocket + HTTP fallback ({POLL_INTERVAL}s)")

    asyncio.run(hybrid_loop())


if __name__ == "__main__":
    main()
