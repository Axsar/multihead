"""Tests for the inbox action layer (multi-scope, key prefixes, deposit helper)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    ScopeType,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore


@pytest.fixture()
def ks(tmp_path: Path) -> KnowledgeStore:
    db = tmp_path / "test_inbox.db"
    store = KnowledgeStore(db)
    return store


def _make_claim(
    ks: KnowledgeStore,
    key: str,
    scope_id: str = "multihead",
    claim_type: ClaimType = ClaimType.PLAN,
    produced_by: str = "other-agent",
    valid_to: datetime | None = None,
) -> Claim:
    now = datetime.now(timezone.utc)
    claim = Claim(
        claim_status=ClaimStatus.ACCEPTED,
        claim_type=claim_type,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=scope_id,
            valid_from=now,
            valid_to=valid_to,
        ),
        canonical=ClaimCanonical(
            claim_key=key,
            subject=EntityRef(entity_type="component", entity_id="test"),
            predicate="has_state",
            object=ValueObject(value_type="string", value=True),
        ),
        statement=f"Test claim: {key}",
        confidence=0.9,
        provenance=Provenance(
            produced_by={"kind": "external", "id": produced_by},
        ),
    )
    return ks.insert_claim(claim)


class TestMultiScopeInbox:
    def test_none_scope_returns_all_scopes(self, ks: KnowledgeStore):
        _make_claim(ks, "action.h2v.work_order.test1", scope_id="h2v")
        _make_claim(ks, "action.multihead.work_order.test2", scope_id="multihead")
        _make_claim(ks, "action.bubble_fill.consensus.test3", scope_id="bubble_fill")

        results = ks.get_unhandled_claims(
            agent_id="me",
            scope_id=None,
            key_prefixes=["action."],
        )
        scopes = {c.scope.scope_id for c in results}
        assert scopes == {"h2v", "multihead", "bubble_fill"}

    def test_scope_filter_limits_results(self, ks: KnowledgeStore):
        _make_claim(ks, "action.h2v.work_order.test1", scope_id="h2v")
        _make_claim(ks, "action.multihead.work_order.test2", scope_id="multihead")

        results = ks.get_unhandled_claims(
            agent_id="me",
            scope_id="h2v",
            key_prefixes=["action."],
        )
        assert len(results) == 1
        assert results[0].scope.scope_id == "h2v"


class TestKeyPrefixFilter:
    def test_key_prefixes_match_action_claims(self, ks: KnowledgeStore):
        _make_claim(ks, "action.multihead.work_order.inbox", claim_type=ClaimType.PLAN)
        _make_claim(ks, "doc.h2v.some_fact", claim_type=ClaimType.FACT)

        results = ks.get_unhandled_claims(
            agent_id="me",
            key_prefixes=["action."],
        )
        assert len(results) == 1
        assert results[0].canonical.claim_key == "action.multihead.work_order.inbox"

    def test_key_prefixes_or_claim_types(self, ks: KnowledgeStore):
        """key_prefixes and claim_types are OR'd — either match surfaces items."""
        _make_claim(ks, "action.multihead.work_order.test", claim_type=ClaimType.PLAN)
        _make_claim(ks, "random.question", claim_type=ClaimType.QUESTION)
        _make_claim(ks, "doc.unrelated", claim_type=ClaimType.FACT)

        results = ks.get_unhandled_claims(
            agent_id="me",
            claim_types=["question"],
            key_prefixes=["action."],
        )
        keys = {c.canonical.claim_key for c in results}
        assert "action.multihead.work_order.test" in keys
        assert "random.question" in keys
        assert "doc.unrelated" not in keys

    def test_multiple_prefixes(self, ks: KnowledgeStore):
        _make_claim(ks, "action.h2v.vote.test")
        _make_claim(ks, "solve.consensus.bezier")
        _make_claim(ks, "doc.unrelated")

        results = ks.get_unhandled_claims(
            agent_id="me",
            key_prefixes=["action.", "solve.consensus."],
        )
        keys = {c.canonical.claim_key for c in results}
        assert "action.h2v.vote.test" in keys
        assert "solve.consensus.bezier" in keys
        assert "doc.unrelated" not in keys


class TestDepositActionClaim:
    def test_deposit_action_enforces_key_pattern(self, ks: KnowledgeStore):
        from multihead.mcp_server._tools_core import _deposit_action_claim

        with patch("multihead.mcp_server._tools_core._get_ks", return_value=ks):
            result = asyncio.get_event_loop().run_until_complete(
                _deposit_action_claim(
                    scope_id="multihead",
                    action_type="work_order",
                    short_id="test-task",
                    statement="Test work order",
                    deadline_hours=24,
                )
            )
        data = json.loads(result)
        assert data["claim_key"] == "action.multihead.work_order.test-task"
        assert "deadline" in data

    def test_deposit_action_rejects_invalid_type(self, ks: KnowledgeStore):
        from multihead.mcp_server._tools_core import _deposit_action_claim

        with patch("multihead.mcp_server._tools_core._get_ks", return_value=ks):
            result = asyncio.get_event_loop().run_until_complete(
                _deposit_action_claim(
                    scope_id="multihead",
                    action_type="invalid_type",
                    short_id="test",
                    statement="Bad",
                )
            )
        assert "Error" in result
        assert "invalid_type" in result
