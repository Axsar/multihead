"""Main narrative pipeline orchestrator.

Ties together: extraction → fusion → verification → state tracking.
This is the entry point for feeding verified, cited claims into
the MultiHead knowledge store.

Usage:
    from multihead.narrative.pipeline import NarrativePipeline
    from multihead.knowledge_store import KnowledgeStore

    store = KnowledgeStore(Path("data/knowledge.db"))
    pipeline = NarrativePipeline(store, project_id="multihead")

    # Ingest git history
    pipeline.ingest_git(Path("/path/to/repo"), since=yesterday)

    # Ingest a chat session
    pipeline.ingest_chat(Path("session.jsonl"), session_id="abc123")

    # Ingest agent results
    pipeline.ingest_agent_result(result_dict, agent_id="qwen-vlm-01")

    # Run fusion + verification on pending evidence
    fused = pipeline.fuse_pending()

    # Apply to persistent state
    pipeline.update_state(fused)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.artifact_store import ArtifactStore
from multihead.knowledge_models import Claim, ClaimStatus, KnowledgeEvent, Record
from multihead.knowledge_store import KnowledgeStore
from multihead.narrative.confidence import SourcePriority
from multihead.narrative.fusion import FusionBundle, FusedNarrative, NarrativeFusion
from multihead.narrative.source_extractors.agent_extractor import AgentExtractor
from multihead.narrative.source_extractors.chat_extractor import ChatExtractor
from multihead.narrative.source_extractors.git_extractor import GitExtractor
from multihead.narrative.source_extractors.markdown_extractor import MarkdownExtractor
from multihead.narrative.claude_enhancer import ClaudeEnhancer
from multihead.narrative.state_engine import NarrativeState, NarrativeStateEngine
from multihead.narrative.verification import NarrativeVerifier

logger = logging.getLogger(__name__)


class NarrativePipeline:
    """Main entry point for the narrative pipeline.

    Orchestrates: extraction → storage → fusion → verification → state.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        project_id: str = "default",
        artifact_store: ArtifactStore | None = None,
    ):
        self.store = store
        self.project_id = project_id
        self.artifact_store = artifact_store

        # Extractors
        self.git_extractor: GitExtractor | None = None
        self.chat_extractor = ChatExtractor(project_id=project_id, artifact_store=artifact_store)
        self.agent_extractor = AgentExtractor(project_id=project_id, artifact_store=artifact_store)
        self.markdown_extractor = MarkdownExtractor(project_id=project_id, artifact_store=artifact_store)

        # Pipeline stages
        self.fusion = NarrativeFusion()
        self.verifier = NarrativeVerifier()
        self.state_engine = NarrativeStateEngine()

        # Pending artifacts (pre-fusion buffer)
        self._pending_records: list[Record] = []
        self._pending_events: list[KnowledgeEvent] = []
        self._pending_claims: list[Claim] = []
        self._pending_sources: dict[str, SourcePriority] = {}

        # Persistent state (loaded/saved externally)
        self.state = NarrativeState()

    # ------------------------------------------------------------------
    # Ingestion methods (extract → store → buffer for fusion)
    # ------------------------------------------------------------------

    def ingest_git(
        self,
        repo_path: Path,
        since: datetime | None = None,
        limit: int = 100000,
    ) -> int:
        """Ingest git commits from a repository.

        Returns number of commits ingested.
        """
        self.git_extractor = GitExtractor(repo_path, self.project_id, artifact_store=self.artifact_store)
        artifacts = self.git_extractor.extract_commits(since=since, limit=limit)

        count = 0
        for artifact in artifacts:
            self._store_and_buffer(artifact, SourcePriority.GIT_COMMIT_MESSAGE)
            count += 1

        logger.info("Ingested %d git commits from %s", count, repo_path.name)
        return count

    def ingest_chat(
        self,
        transcript_path: Path,
        session_id: str | None = None,
    ) -> int:
        """Ingest a chat transcript (JSONL format).

        Returns number of narrative artifacts extracted.
        """
        artifacts = self.chat_extractor.extract_from_jsonl(
            transcript_path, session_id=session_id,
        )

        count = 0
        for artifact in artifacts:
            self._store_and_buffer(artifact, SourcePriority.CHAT_MESSAGE_AUTHOR)
            count += 1

        logger.info(
            "Ingested %d artifacts from chat %s",
            count, transcript_path.name,
        )
        return count

    def ingest_chat_messages(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        source_uri: str = "",
    ) -> int:
        """Ingest chat messages from memory (live session).

        Returns number of narrative artifacts extracted.
        """
        artifacts = self.chat_extractor.extract_from_messages(
            messages, session_id, source_uri=source_uri,
        )

        count = 0
        for artifact in artifacts:
            self._store_and_buffer(artifact, SourcePriority.CHAT_MESSAGE_AUTHOR)
            count += 1

        return count

    def ingest_markdown(
        self,
        doc_path: Path,
        doc_type: str = "plan",
        source_project: str | None = None,
    ) -> int:
        """Ingest a markdown planning/status document.

        Returns number of claims extracted.
        """
        artifacts = self.markdown_extractor.extract_from_file(
            doc_path, doc_type=doc_type, source_project=source_project,
        )

        count = 0
        for artifact in artifacts:
            self._store_and_buffer(artifact, SourcePriority.PLANNING_DOCUMENT)
            count += len(artifact.get("claims", []))

        logger.info(
            "Ingested %d claims from markdown %s", count, doc_path.name,
        )
        return count

    async def ingest_markdown_enhanced(
        self,
        doc_path: Path,
        doc_type: str = "plan",
        source_project: str | None = None,
        acp_url: str | None = None,
        api_key: str | None = None,
        synthesize: bool = True,
    ) -> int:
        """Ingest markdown with Claude-enhanced extraction (Phase 1 + Phase 2).

        Phase 1: Heuristic extraction via MarkdownExtractor.
        Phase 2: Deep semantic extraction via Claude worker daemons.

        Returns total number of merged claims.
        """
        import os
        _acp_url = acp_url or os.environ.get("ACP_URL", "http://localhost:8000/api/v1")
        _api_key = api_key or os.environ.get("ACP_CLAUDE_SESSION_KEY") or os.environ.get("ACP_SESSION_KEY", "")

        if not _api_key:
            logger.warning("No ACP API key — falling back to heuristic-only")
            return self.ingest_markdown(doc_path, doc_type, source_project)

        # Phase 1: Heuristic
        heuristic_artifacts = self.markdown_extractor.extract_from_file(
            doc_path, doc_type=doc_type, source_project=source_project,
        )
        heuristic_claims: list = []
        for art in heuristic_artifacts:
            heuristic_claims.extend(art.get("claims", []))

        logger.info("Phase 1 (heuristic): %d claims from %s", len(heuristic_claims), doc_path.name)

        # Phase 2: Claude enhancement
        enhancer = ClaudeEnhancer(
            acp_url=_acp_url,
            api_key=_api_key,
            project_id=source_project or self.project_id,
        )
        enhanced_artifacts = await enhancer.enhance_document(
            doc_path,
            doc_type=doc_type,
            source_project=source_project,
            heuristic_claims=heuristic_claims,
            synthesize=synthesize,
        )

        count = 0
        for artifact in enhanced_artifacts:
            self._store_and_buffer(artifact, SourcePriority.LLM_INFERENCE)
            count += len(artifact.get("claims", []))

        logger.info(
            "Enhanced ingestion: %d total claims from %s (heuristic=%d)",
            count, doc_path.name, len(heuristic_claims),
        )
        return count

    def ingest_agent_result(
        self,
        result: dict[str, Any],
        agent_id: str,
        agent_type: str = "claude",
        parent_session_id: str | None = None,
    ) -> int:
        """Ingest a single agent task result.

        Returns 1 on success, 0 on failure.
        """
        artifact = self.agent_extractor.extract_from_result(
            result, agent_id, agent_type, parent_session_id,
        )

        source = (
            SourcePriority.VLM_INFERENCE if agent_type in ("qwen-vlm", "vlm")
            else SourcePriority.LLM_INFERENCE if agent_type in ("claude", "gpt")
            else SourcePriority.AGENT_INTERPRETATION
        )

        self._store_and_buffer(artifact, source)
        return 1

    # ------------------------------------------------------------------
    # Fusion + Verification
    # ------------------------------------------------------------------

    def fuse_pending(
        self,
        unit_id: str | None = None,
        unit_type: str = "session",
    ) -> FusedNarrative:
        """Fuse all pending evidence into a coherent narrative.

        Returns FusedNarrative with accepted/contested claims.
        """
        bundle = FusionBundle(
            unit_id=unit_id or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            unit_type=unit_type,
            records=list(self._pending_records),
            events=list(self._pending_events),
            claims=list(self._pending_claims),
            claim_sources=dict(self._pending_sources),
        )

        fused = self.fusion.fuse(bundle)

        logger.info(
            "Fusion complete: %d accepted, %d contested, %d conflicts",
            len(fused.accepted_claims),
            len(fused.contested_claims),
            len(fused.conflicts),
        )

        # Clear pending buffer
        self._pending_records.clear()
        self._pending_events.clear()
        self._pending_claims.clear()
        self._pending_sources.clear()

        return fused

    # ------------------------------------------------------------------
    # State Engine
    # ------------------------------------------------------------------

    def update_state(self, narrative: FusedNarrative) -> NarrativeState:
        """Apply a fused narrative to the persistent state.

        Returns updated NarrativeState.
        """
        self.state = self.state_engine.apply(narrative, self.state)

        logger.info(
            "State updated: %d entities, %d sessions processed",
            len(self.state.entities),
            len(self.state.sessions_processed),
        )

        return self.state

    # ------------------------------------------------------------------
    # Convenience: full pipeline in one call
    # ------------------------------------------------------------------

    def run_full(
        self,
        unit_id: str | None = None,
        unit_type: str = "session",
    ) -> FusedNarrative:
        """Fuse pending evidence and apply to state in one call.

        Returns FusedNarrative.
        """
        fused = self.fuse_pending(unit_id=unit_id, unit_type=unit_type)
        self.update_state(fused)
        return fused

    # ------------------------------------------------------------------
    # Store integration
    # ------------------------------------------------------------------

    def store_fused_claims(self, narrative: FusedNarrative) -> int:
        """Persist accepted claims to the KnowledgeStore.

        Returns number of claims stored.
        """
        count = 0
        for claim in narrative.accepted_claims:
            try:
                claim.claim_status = ClaimStatus.ACCEPTED

                # Insert evidence pointers first (they may reference records that must exist)
                for ep in claim.evidence_supports:
                    try:
                        self.store.insert_evidence(ep)
                    except Exception as e:
                        # Log details if FK constraint fails
                        logger.debug("Evidence insert: %s (record_id=%s)", e, ep.record_id)

                # Now insert the claim
                self.store.insert_claim(claim)

                # Link evidence to claim
                for ep in claim.evidence_supports:
                    self.store.link_claim_evidence(
                        claim.claim_id, ep.evidence_id, "supports",
                    )
                count += 1
            except Exception as e:
                logger.warning("Failed to store claim %s: %s", claim.claim_id, e)

        logger.info("Stored %d claims to knowledge store", count)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_and_buffer(
        self,
        artifact: dict[str, Any],
        source: SourcePriority,
    ) -> None:
        """Store artifact in knowledge store and buffer for fusion."""
        record = artifact.get("record")
        event = artifact.get("event")
        claims = artifact.get("claims", [])
        evidence_list = artifact.get("evidence", [])

        # Store record
        if record:
            try:
                self.store.insert_record(record)
            except Exception:
                pass  # Record may already exist
            self._pending_records.append(record)

        # Store evidence
        for ep in evidence_list:
            try:
                self.store.insert_evidence(ep)
            except Exception:
                pass

        # Store and buffer event
        if event:
            try:
                stored_event = self.store.insert_event(event)
                # Link evidence to event
                for ep in event.evidence_supports:
                    self.store.link_event_evidence(
                        stored_event.event_id, ep.evidence_id, "supports",
                    )
            except Exception as e:
                logger.debug("Event storage: %s", e)
            self._pending_events.append(event)

        # Buffer claims for fusion (NOT stored yet — stored after fusion)
        # Skip claims that already exist in the store (dedup by claim_key)
        for claim in claims:
            existing = self.store.get_accepted_claim(
                claim.scope.scope_type.value if hasattr(claim.scope.scope_type, 'value') else str(claim.scope.scope_type),
                claim.scope.scope_id,
                claim.canonical.claim_key,
            )
            if existing:
                logger.debug("Skipping duplicate claim: %s", claim.canonical.claim_key)
                continue
            self._pending_claims.append(claim)
            self._pending_sources[claim.claim_id] = source
