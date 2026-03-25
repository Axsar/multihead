"""Night Shift stages 0-7: harvest, input window, chunking, extraction."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from multihead.chunker import Chunk
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EventType,
    KnowledgeEvent,
    ScopeType,
    TimeBlock,
    ValueObject,
)

from multihead.claim_corroboration import corroborate_claim, extract_file_paths
from multihead.scope_inference import infer_scope

from .models import _prov

import re

# Patterns indicating a claim describes a state change (before→after)
_TEMPORAL_PATTERNS = re.compile(
    r'\b(now |was |previously |before |after |fixed |changed |updated |no longer |'
    r'went from |improved |regressed |used to )',
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


class EarlyStagesMixin:
    """Mixin providing Night Shift stages 0 through 7."""

    def _supersede_temporal(self, key_prefix: str, new_claim_id: str) -> int:
        """Supersede older claims with the same key prefix.

        When a new claim describes a state change (e.g. "X is now fixed"),
        find older claims about the same entity and mark them superseded.
        Only supersedes claims that are NOT already superseded/rejected.

        Returns count of superseded claims.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self.knowledge._connect() as conn:
                rows = conn.execute(
                    "SELECT claim_id FROM claims "
                    "WHERE claim_key LIKE ? "
                    "AND claim_status NOT IN ('superseded', 'rejected') "
                    "AND claim_id != ? "
                    "LIMIT 100000",
                    (f"{key_prefix}%", new_claim_id),
                ).fetchall()
                count = 0
                for row in rows:
                    conn.execute(
                        "UPDATE claims SET claim_status = 'superseded', "
                        "superseded_by_claim_id = ?, updated_at = ? "
                        "WHERE claim_id = ?",
                        (new_claim_id, now, row["claim_id"]),
                    )
                    count += 1
                return count
        except Exception:
            return 0

    def _get_last_successful_run_time(self) -> datetime | None:
        """Get the timestamp of the last successful Night Shift run from knowledge store."""
        try:
            # Query for all claims in multihead scope, then filter by key prefix
            claims = self.knowledge.list_claims(
                status="accepted",
                scope_id="multihead",
                limit=100000,
            )
            # Filter for our tracking claims (prefix match)
            matching = [
                c for c in claims
                if c.canonical.claim_key.startswith("nightshift.last_run.")
            ]
            if matching:
                # Most recent claim by provenance timestamp
                latest = max(matching, key=lambda c: c.provenance.created_at)
                # Parse timestamp from object value
                timestamp_str = latest.canonical.object.value
                return datetime.fromisoformat(timestamp_str)
        except Exception as e:
            logger.warning(f"Failed to get last successful run time: {e}")
        return None

    def _update_last_successful_run_time(self, timestamp: datetime) -> None:
        """Update the last successful Night Shift run timestamp in knowledge store."""
        try:
            # Create new claim with timestamp-based key to avoid UNIQUE constraint
            claim_key = f"nightshift.last_run.{int(timestamp.timestamp())}"

            claim = Claim(
                claim_type=ClaimType.FACT,
                scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
                canonical=ClaimCanonical(
                    claim_key=claim_key,
                    subject=EntityRef(
                        entity_type="pipeline",
                        entity_id="nightshift",
                        entity_label="Night Shift",
                    ),
                    predicate="completed_at",
                    object=ValueObject(value_type="datetime", value=timestamp.isoformat()),
                ),
                statement=f"Night Shift completed at {timestamp.isoformat()}",
                provenance=_prov(),
            )
            claim.claim_status = ClaimStatus.ACCEPTED  # Auto-accept this tracking claim
            self.knowledge.insert_claim(claim)

            logger.info(f"Updated last successful run timestamp to {timestamp.isoformat()}")
        except Exception as e:
            logger.warning(f"Failed to update last successful run time: {e}")

    async def _stage_session_harvest(self, context: dict) -> dict:
        """Stage 0: Harvest knowledge from all Claude Code session files.

        Two harvesters run in sequence:
        1. SessionHarvester — MEMORY.md / CLAUDE.md files → direct claims
        2. ConversationHarvester — JSONL session transcripts → RecordStore
           (downstream stages chunk and extract from these records)
        """
        from multihead.session_harvester import SessionHarvester
        from multihead.conversation_harvester import ConversationHarvester

        data_dir = self.output_dir  # nightshift/ dir itself — manifests stored here

        # 1. Memory file harvest (existing)
        harvester = SessionHarvester(
            knowledge_store=self.knowledge,
            data_dir=data_dir,
        )
        result = harvester.harvest_all(on_progress=self._emit)

        # 2. Conversation transcript harvest (new)
        conv_harvester = ConversationHarvester(
            record_store=self.records,
            knowledge_store=self.knowledge,
            data_dir=data_dir,
        )
        conv_result = conv_harvester.harvest_all()

        return {
            "metrics": {
                "projects_scanned": result.projects_scanned,
                "projects_harvested": result.projects_harvested,
                "claims_deposited": result.claims_deposited,
                "evolution_events": result.evolution_events,
                "conv_files_processed": conv_result.files_processed,
                "conv_exchanges_ingested": conv_result.exchanges_ingested,
                "conv_records_created": conv_result.records_created,
            },
            "summary": (
                f"{result.projects_scanned} projects scanned, "
                f"{result.claims_deposited} claims, "
                f"{result.evolution_events} evolution events, "
                f"{conv_result.files_processed} transcripts → "
                f"{conv_result.records_created} records"
            ),
        }

    async def _stage_select_input_window(self, context: dict) -> dict | None:
        # Try to get the last successful run time
        last_run = self._get_last_successful_run_time()

        if last_run:
            # Process all records since last successful run
            since = last_run
            logger.info(f"Using last successful run time: {since.isoformat()}")
        else:
            # First run: get ALL records, not just last 24h
            since = datetime(2020, 1, 1, tzinfo=timezone.utc)
            logger.info("No previous run found — processing all records")

        count = self.records.count_new_records(since)
        if count == 0:
            logger.info("No new conversation records — stages 8-10 (code/git/CI) will still run")
            # Don't return None — independent stages (behavioral, git, CI) don't need records
        # Also count content records (exclude presence pings with no sha256)
        records = self.records.get_records_since(since)
        content_count = sum(1 for r in records if r.sha256)
        context["since"] = since
        return {"since": since, "record_count": content_count, "total_records": count}

    async def _stage_normalize_chunk(self, context: dict) -> dict:
        since = context.get("since", datetime.now(timezone.utc) - timedelta(hours=24))
        records = self.records.get_records_since(since)

        all_chunks: list[Chunk] = []
        total_chars = 0
        parsed_chars = 0
        seen_sha: set[str] = set()  # Deduplicate by content hash

        for record in records:
            # Skip records without content (e.g. presence pings)
            if not record.sha256:
                continue
            # Skip duplicate content (same file ingested multiple times)
            if record.sha256 in seen_sha:
                continue
            seen_sha.add(record.sha256)
            content = self.records.get_content(record)
            if content is None:
                continue
            text_len = len(content)
            total_chars += text_len
            try:
                chunks = self.chunker.chunk_record(record, content)
                all_chunks.extend(chunks)
                parsed_chars += text_len
            except Exception as e:
                logger.warning(f"Failed to chunk record {record.record_id}: {e}")

        # Filter out very short chunks — not worth LLM calls
        min_chars = 200
        filtered = [c for c in all_chunks if len(c.text) >= min_chars]
        skipped = len(all_chunks) - len(filtered)
        if skipped:
            logger.info(f"Filtered {skipped} chunks < {min_chars} chars (kept {len(filtered)})")

        # Optional cap on chunk count (for faster end-to-end runs)
        max_chunks = context.get("max_chunks")
        if max_chunks and len(filtered) > max_chunks:
            logger.info(f"Capping chunks from {len(filtered)} to {max_chunks}")
            filtered = filtered[:max_chunks]

        success_rate = parsed_chars / total_chars if total_chars else 1.0
        coverage = self.chunker.compute_coverage(all_chunks, total_chars) if total_chars else 1.0

        return {
            "chunks": filtered,
            "metrics": {
                "parse_success_rate": success_rate,
                "chunk_coverage": coverage,
                "chunk_count": len(filtered),
                "chunks_filtered": skipped,
            },
        }

    async def _stage_hot_signals(self, context: dict) -> dict:
        """Compute recency/frequency signals without LLM."""
        chunks: list[Chunk] = context.get("chunks", [])
        word_freq: Counter[str] = Counter()
        for chunk in chunks:
            words = chunk.text.lower().split()
            word_freq.update(words)

        # Top entities by frequency (simple heuristic)
        top_words = word_freq.most_common(50)
        return {
            "hot_signals": {
                "top_terms": [{"term": w, "count": c} for w, c in top_words],
            },
            "metrics": {},
        }

    def _make_chunk_progress_cb(self, stage_name: str, chunks: list):
        """Build a callback that emits chunk_progress events.

        The callback receives (chunk_index, chunk_total) from the extractor.
        Note: chunk_index may not correspond 1:1 to entries in *chunks*
        (e.g. TopicAssigner processes batches), so we only report the
        numeric progress rather than indexing into the list.
        """

        def _cb(chunk_index: int, chunk_total: int) -> None:
            self._emit({
                "event": "chunk_progress",
                "stage": stage_name,
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
            })

        return _cb

    async def _adapter_or_fn(self):
        """Return raw adapter for batch mode, generate closure otherwise."""
        if self.config.batch_mode:
            return await self._get_extraction_adapter()
        return self._generate_fn()

    async def _stage_entity_extraction(self, context: dict) -> dict:
        chunks = context.get("chunks", [])
        self._emit({"event": "file_start", "stage": "entity_extraction",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "source_project": "", "chunk_total": len(chunks)})
        result = await self.entity_extractor.extract(
            chunks, await self._adapter_or_fn(), concurrency=self.config.concurrency,
            checkpoint_dir=self.output_dir, stage_name="entity_extraction",
            batch_mode=self.config.batch_mode, no_wait=self.config.no_wait,
            on_chunk_progress=self._make_chunk_progress_cb("entity_extraction", chunks),
        )
        self._emit({"event": "file_done", "stage": "entity_extraction",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "chunk_count": len(chunks), "results_count": len(result.items)})
        return {"entities": result.items, "metrics": result.metrics}

    async def _stage_topic_assignment(self, context: dict) -> dict:
        chunks = list(context.get("chunks", []))  # COPY — don't mutate shared context

        # Also include code/behavioral/git claims as pseudo-chunks for topic assignment
        from multihead.chunker import Chunk
        with self.knowledge._connect() as conn:
            code_rows = conn.execute(
                "SELECT claim_id, statement, observation_method FROM claims "
                "WHERE observation_method IN ('code_read', 'code_behavior_llm', 'git_diff', 'ci_test', 'test_result') "
                "AND claim_status NOT IN ('superseded', 'rejected') "
                "AND length(statement) > 50"
            ).fetchall()

        if code_rows:
            logger.info("Topic assignment: adding %d code/git claims as pseudo-chunks", len(code_rows))
            for row in code_rows:
                chunks.append(Chunk(
                    chunk_id=f"claim_{row['claim_id']}",
                    text=f"[{row['observation_method']}] {row['statement']}",
                    record_id="",
                    span_start=0,
                    span_end=0,
                ))

        self._emit({"event": "file_start", "stage": "topic_assignment",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "source_project": "", "chunk_total": len(chunks)})
        result = await self.topic_assigner.extract(
            chunks, await self._adapter_or_fn(), concurrency=self.config.concurrency,
            checkpoint_dir=self.output_dir, stage_name="topic_assignment",
            batch_mode=self.config.batch_mode, no_wait=self.config.no_wait,
            on_chunk_progress=self._make_chunk_progress_cb("topic_assignment", chunks),
        )
        self._emit({"event": "file_done", "stage": "topic_assignment",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "chunk_count": len(chunks), "results_count": len(result.items)})
        return {"topics": result.items, "metrics": result.metrics}

    async def _stage_event_extraction(self, context: dict) -> dict:
        chunks = context.get("chunks", [])
        self._emit({"event": "file_start", "stage": "event_extraction",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "source_project": "", "chunk_total": len(chunks)})
        result = await self.event_extractor.extract(
            chunks, await self._adapter_or_fn(), concurrency=self.config.concurrency,
            checkpoint_dir=self.output_dir, stage_name="event_extraction",
            batch_mode=self.config.batch_mode, no_wait=self.config.no_wait,
            on_chunk_progress=self._make_chunk_progress_cb("event_extraction", chunks),
        )

        # Build chunk_id → agent_folder lookup
        chunk_agent: dict[str, str] = {}
        record_ids = {c.record_id for c in chunks if c.record_id}
        if record_ids:
            with self.knowledge._connect() as conn:
                placeholders = ",".join("?" for _ in record_ids)
                rows = conn.execute(
                    f"SELECT record_id, uri FROM records WHERE record_id IN ({placeholders})",
                    list(record_ids),
                ).fetchall()
                for r in rows:
                    uri = r["uri"]
                    if "://" in uri:
                        parts = uri.split("://", 1)[1].split("/")
                        if parts:
                            # Map all chunks from this record to the agent folder
                            for c in chunks:
                                if c.record_id == r["record_id"]:
                                    chunk_agent[c.chunk_id] = parts[0]

        # Store extracted events in knowledge store
        events_created = 0
        for evt_data in result.items:
            try:
                chunk_id = evt_data.get("source_chunk_id", "")
                agent_folder = chunk_agent.get(chunk_id, "")
                evt = KnowledgeEvent(
                    event_type=EventType(evt_data.get("event_type", "note")),
                    title=evt_data.get("title", "Untitled event")[:160],
                    summary=evt_data.get("summary", ""),
                    time=TimeBlock(happened_at=datetime.now(timezone.utc)),
                    provenance=_prov(
                        source_anchor={
                            k: v for k, v in {
                                "source_chunk_id": chunk_id,
                                "agent_folder": agent_folder,
                            }.items() if v
                        },
                    ),
                )
                self.knowledge.insert_event(evt)
                events_created += 1
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed event: {e}")

        self._emit({"event": "file_done", "stage": "event_extraction",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "chunk_count": len(chunks), "results_count": len(result.items)})

        return {
            "extracted_events": result.items,
            "events_created": events_created,
            "metrics": result.metrics,
        }

    async def _stage_claim_extraction(self, context: dict) -> dict:
        chunks = context.get("chunks", [])
        self._emit({"event": "file_start", "stage": "claim_extraction",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "source_project": "", "chunk_total": len(chunks)})
        result = await self.claim_extractor.extract(
            chunks, await self._adapter_or_fn(), concurrency=self.config.concurrency,
            checkpoint_dir=self.output_dir, stage_name="claim_extraction",
            batch_mode=self.config.batch_mode, no_wait=self.config.no_wait,
            on_chunk_progress=self._make_chunk_progress_cb("claim_extraction", chunks),
        )
        self._emit({"event": "file_done", "stage": "claim_extraction",
                     "file_name": "chunks", "file_index": 0, "file_total": 1,
                     "chunk_count": len(chunks), "results_count": len(result.items)})

        # Build chunk_id → (session_id, session_date) lookup for session dating
        chunk_session: dict[str, tuple[str, str]] = {}
        record_ids = {c.record_id for c in chunks if c.record_id}
        if record_ids:
            with self.knowledge._connect() as conn:
                placeholders = ",".join("?" for _ in record_ids)
                rows = conn.execute(
                    f"SELECT record_id, uri, captured_at FROM records WHERE record_id IN ({placeholders})",
                    list(record_ids),
                ).fetchall()
                record_map = {r["record_id"]: (r["uri"], r["captured_at"]) for r in rows}
            for chunk in chunks:
                if chunk.record_id in record_map:
                    uri, captured = record_map[chunk.record_id]
                    # Extract project + session from URI: conversation://PROJECT_NAME/SESSION_ID/turn/N
                    session_id = ""
                    agent_folder = ""
                    if "://" in uri:
                        parts = uri.split("://", 1)[1].split("/")
                        if len(parts) >= 2:
                            agent_folder = parts[0]  # project folder name
                            session_id = parts[1]    # session UUID
                        elif parts:
                            session_id = parts[0]
                    # Extract date from captured_at
                    session_date = captured[:10] if captured else ""
                    chunk_session[chunk.chunk_id] = (session_id, session_date, agent_folder)

        # Build all claims first (no DB writes yet)
        claims_to_insert: list[Claim] = []
        skipped_short = 0
        skipped_malformed = 0
        total_items = len(result.items)
        for idx, claim_data in enumerate(result.items):
            stmt = claim_data.get("statement", "")
            if len(stmt.strip()) < 50:
                skipped_short += 1
                continue
            try:
                # Map non-standard claim types to valid ones
                raw_type = claim_data.get("claim_type", "fact")
                type_map = {
                    "architecture": "fact",
                    "outcome": "fact",
                    "observation": "fact",
                    "implementation": "fact",
                    "requirement": "constraint",
                    "goal": "plan",
                    "finding": "fact",
                }
                claim_type_str = type_map.get(raw_type, raw_type)

                claim = Claim(
                    claim_type=ClaimType(claim_type_str),
                    scope=ClaimScope(
                        scope_type=ScopeType.PROJECT,
                        scope_id=claim_data.get("scope_id") or infer_scope(
                            claim_data.get("claim_key", ""),
                            stmt,
                        ),
                    ),
                    canonical=ClaimCanonical(
                        claim_key=claim_data.get("claim_key", "unknown.key"),
                        subject=EntityRef(
                            entity_type=claim_data.get("entity_type", "concept"),
                            entity_id=claim_data.get("entity_id", "unknown"),
                        ),
                        predicate=claim_data.get("predicate", "is"),
                        object=ValueObject(
                            value_type="string",
                            value=claim_data.get("object_value", ""),
                        ),
                    ),
                    statement=stmt[:4000],
                    rationale=claim_data.get("rationale", ""),
                    confidence=claim_data.get("confidence", 0.5),
                    provenance=_prov(
                        observation_method=claim_data.get("observation_method", "conversation_analysis"),
                        speaker=claim_data.get("speaker", ""),
                        valence=claim_data.get("valence", ""),
                        source_anchor={
                            k: v for k, v in {
                                "file_path": claim_data.get("file_path", ""),
                                "symbol": claim_data.get("symbol", ""),
                                "source_chunk_id": claim_data.get("source_chunk_id", ""),
                                "session_id": chunk_session.get(claim_data.get("source_chunk_id", ""), ("", "", ""))[0],
                                "session_date": chunk_session.get(claim_data.get("source_chunk_id", ""), ("", "", ""))[1],
                                "agent_folder": chunk_session.get(claim_data.get("source_chunk_id", ""), ("", "", ""))[2],
                            }.items() if v
                        },
                        evidence=[e for e in claim_data.get("evidence", []) if isinstance(e, dict)],
                    ),
                )
                claims_to_insert.append(claim)
            except (ValueError, KeyError) as e:
                skipped_malformed += 1
                if skipped_malformed <= 5:
                    logger.warning("Skipping malformed claim: %s", e)

            if (idx + 1) % 1000 == 0:
                logger.info("Parsed %d/%d items (%d claims, %d short, %d malformed)",
                            idx + 1, total_items, len(claims_to_insert), skipped_short, skipped_malformed)

        logger.info("Parsed all %d items: %d claims to insert, %d short, %d malformed",
                     total_items, len(claims_to_insert), skipped_short, skipped_malformed)

        # Batch embed all claims at once (instead of per-claim)
        if claims_to_insert:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                model = SentenceTransformer("all-MiniLM-L6-v2")
                statements = [c.statement for c in claims_to_insert]
                logger.info("Batch embedding %d claims...", len(statements))
                embeddings = model.encode(statements, batch_size=512, normalize_embeddings=True, show_progress_bar=False)
                logger.info("Embeddings computed in batch")
            except ImportError:
                embeddings = None
                logger.info("sentence-transformers not available, skipping batch embed")

        # Bulk insert: dedup=False (skip expensive per-claim FTS check)
        claims_created = 0
        for idx, claim in enumerate(claims_to_insert):
            try:
                self.knowledge.insert_claim(claim, dedup=False)
                claims_created += 1
            except Exception as e:
                if "UNIQUE" not in str(e):
                    logger.debug("Insert error: %s", e)

            if (idx + 1) % 500 == 0:
                logger.info("Inserted %d/%d claims", idx + 1, len(claims_to_insert))

        # Bulk insert embeddings into claims_vec
        if claims_to_insert and embeddings is not None:
            logger.info("Inserting %d embeddings into claims_vec...", len(claims_to_insert))
            try:
                import sqlite_vec
                with self.knowledge._connect() as conn:
                    sqlite_vec.load(conn)
                    for idx, (claim, emb) in enumerate(zip(claims_to_insert, embeddings)):
                        row = conn.execute(
                            "SELECT rowid FROM claims WHERE claim_id = ?", (claim.claim_id,)
                        ).fetchone()
                        if row:
                            conn.execute(
                                "INSERT OR REPLACE INTO claims_vec(rowid, embedding) VALUES (?, ?)",
                                [row[0], emb.astype(np.float32).tobytes()],
                            )
                        if (idx + 1) % 1000 == 0:
                            logger.info("Embedded %d/%d", idx + 1, len(claims_to_insert))
                logger.info("All embeddings inserted")
            except Exception as e:
                logger.warning("Bulk embedding insert failed: %s", e)

        if skipped_short:
            logger.info("Skipped %d fragment claims (<50 chars)", skipped_short)
        if skipped_malformed:
            logger.info("Skipped %d malformed claims", skipped_malformed)

        return {
            "extracted_claims": result.items,
            "claims_created": context.get("claims_created", 0) + claims_created,
            "claims_skipped_short": skipped_short,
            "claims_skipped_malformed": skipped_malformed,
            "metrics": result.metrics,
        }
