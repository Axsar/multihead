#!/usr/bin/env python3
"""
Basic MultiHead usage examples.

Run this script to see how to:
1. Connect to MultiHead
2. Chat with the local LLM
3. Write and read from the knowledge store
4. Check model head status

Prerequisites:
- MultiHead daemon running: multihead serve
- Virtual environment activated: source .venv/bin/activate

Usage:
    python examples/basic_usage.py
"""

from multihead.client import MultiHeadClient


def main():
    # Connect to MultiHead (defaults to localhost:7337)
    mh = MultiHeadClient()

    # Check connection
    print("🔌 Connecting to MultiHead...")
    if not mh.ping():
        print("❌ MultiHead daemon not running. Start it with: multihead serve")
        return
    print("✅ Connected!\n")

    # Example 1: Chat with the local LLM
    print("=" * 60)
    print("Example 1: Chat with the local LLM")
    print("=" * 60)
    response = mh.chat("What is the capital of France?")
    print(f"Question: What is the capital of France?")
    print(f"Answer: {response['response']}\n")

    # Example 2: Write to the knowledge store
    print("=" * 60)
    print("Example 2: Write to the knowledge store")
    print("=" * 60)
    mh.deposit_claim(
        claim_key="example.greeting.status",
        statement="Successfully ran basic_usage.py example",
        produced_by="basic_usage_script",
        claim_type="fact",
        confidence=1.0,
    )
    print("✅ Deposited claim: 'Successfully ran basic_usage.py example'\n")

    # Example 3: Read from the knowledge store
    print("=" * 60)
    print("Example 3: Read from the knowledge store")
    print("=" * 60)
    claims = mh.query_claims(claim_key="example.greeting", limit=5)
    print(f"Found {len(claims)} claims matching 'example.greeting':")
    for claim in claims:
        print(f"  - {claim['statement']}")
    print()

    # Example 4: Report an event
    print("=" * 60)
    print("Example 4: Report an event")
    print("=" * 60)
    mh.report_event(
        title="Basic usage script completed",
        summary="Ran all 4 examples successfully",
        event_type="task_completed",
        produced_by="basic_usage_script",
        metrics={"examples_run": 4},
    )
    print("✅ Reported event: 'Basic usage script completed'\n")

    # Example 5: Check model head status
    print("=" * 60)
    print("Example 5: Check model head status")
    print("=" * 60)
    heads = mh.list_heads()
    print(f"Available model heads ({len(heads)}):")
    for head in heads:
        status = head.get("state", "unknown")
        model = head.get("model", "unknown")
        kind = head.get("kind", "unknown")
        print(f"  - {head['head_id']}: {model} ({kind}) - {status}")
    print()

    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - See docs/12-getting-started.md for more features")
    print("  - Explore config/recipes/ for pipeline examples")
    print("  - Try multihead chat for interactive mode")
    print("  - Use MCP tools from Claude Code")


if __name__ == "__main__":
    main()
