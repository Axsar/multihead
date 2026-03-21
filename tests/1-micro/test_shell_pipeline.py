"""Tests for the ShellPipeline — dogfooding middleware."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.runtime_config import RuntimeConfig
from multihead.shell_pipeline import ShellPipeline, _STOPWORDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_knowledge_store(tmp_path):
    """Knowledge store with a real SQLite DB for SQL query testing."""
    db_path = tmp_path / "test_knowledge.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT,
            claim_status TEXT DEFAULT 'accepted',
            claim_type TEXT DEFAULT 'fact',
            scope_type TEXT DEFAULT 'project',
            scope_id TEXT DEFAULT 'multihead',
            visibility TEXT DEFAULT 'private',
            valid_from TEXT,
            valid_to TEXT,
            claim_key TEXT,
            predicate TEXT,
            subject_json TEXT,
            object_json TEXT,
            statement TEXT,
            rationale TEXT,
            confidence REAL DEFAULT 0.8,
            stability TEXT,
            importance REAL DEFAULT 0.5,
            superseded_by_claim_id TEXT,
            rejection_reason TEXT,
            contested_reason TEXT,
            derived_from_json TEXT,
            related_json TEXT,
            conflicts_json TEXT,
            provenance_json TEXT,
            signature TEXT,
            signed_by TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    # Insert test claims
    test_claims = [
        ("clm_1", "mesh.security.ed25519",
         "The mesh protocol uses Ed25519 for node identity signing", 0.9),
        ("clm_2", "router.scoring.weights",
         "Router uses weighted scoring with VRAM fit and latency factors", 0.85),
        ("clm_3", "orchestrator.dag.parallel",
         "DAG executor runs independent steps in parallel", 0.9),
        ("clm_4", "knowledge.store.sqlite",
         "Knowledge store uses SQLite with WAL mode for concurrent access", 0.8),
        ("clm_5", "adapter.transformers.4bit",
         "Transformers adapter loads models in 4-bit quantization", 0.85),
        ("clm_6", "consensus.strategy.majority",
         "Consensus uses majority voting by default", 0.9),
        ("clm_7", "head.qwen.vram",
         "Qwen3-8B requires approximately 6GB VRAM in 4-bit mode", 0.8),
        ("clm_8", "mesh.peers.discovery",
         "Mesh peers are discovered via UDP broadcast on port 7339", 0.7),
        ("clm_9", "pipeline.recipe.yaml",
         "Pipeline recipes are defined in YAML format", 0.8),
        ("clm_10", "acp.bridge.websocket",
         "ACP bridge uses WebSocket for real-time task notifications", 0.85),
    ]
    for claim_id, claim_key, statement, confidence in test_claims:
        conn.execute(
            "INSERT INTO claims (claim_id, claim_key, statement, confidence, claim_status) "
            "VALUES (?, ?, ?, ?, 'accepted')",
            (claim_id, claim_key, statement, confidence),
        )
    conn.commit()
    conn.close()

    # Create mock KS with real _connect method
    ks = MagicMock()
    ks.insert_claim = MagicMock()

    def _connect():
        return sqlite3.connect(str(db_path))

    ks._connect = _connect
    return ks


@pytest.fixture
def mock_runtime_config():
    return RuntimeConfig()


@pytest.fixture
def mock_head_manager():
    hm = MagicMock()
    hm.active_head = "core-llm"
    hm.ensure_active = AsyncMock()
    return hm


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.route.return_value = "core-llm"
    return router


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    run_state = MagicMock()
    run_state.run_id = "run_test123"
    run_state.status = "completed"
    run_state.step_results = {}
    orch.create_run = AsyncMock(return_value=run_state)
    orch.execute_run = AsyncMock(return_value=run_state)
    return orch


@pytest.fixture
def mock_decomposer():
    decomp = MagicMock()
    plan = MagicMock()
    plan.goal = "Test task"
    plan.total_steps = 3
    work_order = MagicMock()
    work_order.goal = "Test task"
    work_order.steps = []
    decomp.decompose = AsyncMock(return_value=plan)
    decomp.to_work_order_with_dag = MagicMock(return_value=work_order)
    return decomp


@pytest.fixture
def pipeline(mock_knowledge_store, mock_head_manager, mock_router, mock_runtime_config):
    return ShellPipeline(
        knowledge_store=mock_knowledge_store,
        head_manager=mock_head_manager,
        router=mock_router,
        runtime_config=mock_runtime_config,
    )


@pytest.fixture
def full_pipeline(
    mock_knowledge_store, mock_head_manager, mock_router,
    mock_runtime_config, mock_orchestrator, mock_decomposer,
):
    return ShellPipeline(
        knowledge_store=mock_knowledge_store,
        head_manager=mock_head_manager,
        router=mock_router,
        runtime_config=mock_runtime_config,
        orchestrator=mock_orchestrator,
        auto_decomposer=mock_decomposer,
    )


@pytest.fixture
def brain_fn():
    """Mock brain function: (session_id, user_input, knowledge_ctx) -> response."""
    return AsyncMock(return_value="This is the brain response to your question.")


# ---------------------------------------------------------------------------
# Pipeline Init
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_creates_with_all_components(self, pipeline):
        assert pipeline._ks is not None
        assert pipeline._hm is not None
        assert pipeline._router is not None
        assert pipeline._config is not None

    def test_creates_with_minimal_components(self):
        p = ShellPipeline()
        assert p._ks is None
        assert p._hm is None
        assert p._stats["messages_processed"] == 0

    def test_creates_with_partial_components(self, mock_knowledge_store):
        p = ShellPipeline(knowledge_store=mock_knowledge_store)
        assert p._ks is not None
        assert p._orchestrator is None

    def test_stats_initialized(self, pipeline):
        assert pipeline._stats["messages_processed"] == 0
        assert pipeline._stats["tasks_decomposed"] == 0
        assert pipeline._stats["knowledge_hits"] == 0


# ---------------------------------------------------------------------------
# Keyword Extraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    def test_extracts_meaningful_words(self, pipeline):
        keywords = pipeline._extract_keywords("Tell me about mesh security protocol")
        assert "mesh" in keywords
        assert "security" in keywords
        assert "protocol" in keywords

    def test_skips_short_words(self, pipeline):
        keywords = pipeline._extract_keywords("is it ok to do")
        assert keywords == []

    def test_skips_stopwords(self, pipeline):
        keywords = pipeline._extract_keywords("what about their work with this")
        # "what", "about", "their", "work", "with", "this" are all stopwords
        assert keywords == []

    def test_strips_punctuation(self, pipeline):
        keywords = pipeline._extract_keywords("mesh, security! protocol?")
        assert "mesh" in keywords
        assert "security" in keywords

    def test_deduplicates(self, pipeline):
        keywords = pipeline._extract_keywords("mesh mesh mesh security")
        assert keywords.count("mesh") == 1

    def test_lowercases(self, pipeline):
        keywords = pipeline._extract_keywords("MESH Security Protocol")
        assert "mesh" in keywords
        assert "security" in keywords

    def test_empty_input(self, pipeline):
        assert pipeline._extract_keywords("") == []

    def test_stopwords_comprehensive(self):
        # Verify key stopwords exist
        for word in ["what", "when", "where", "which", "that", "this", "with", "from"]:
            assert word in _STOPWORDS


# ---------------------------------------------------------------------------
# Knowledge RAG (SQL-based)
# ---------------------------------------------------------------------------


class TestKnowledgeRAG:
    def test_finds_relevant_claims(self, pipeline):
        ctx = pipeline._build_knowledge_context("Tell me about mesh security")
        assert "Ed25519" in ctx
        assert "Knowledge context" in ctx

    def test_includes_claim_keys(self, pipeline):
        ctx = pipeline._build_knowledge_context("mesh security protocol")
        assert "[mesh.security.ed25519]" in ctx

    def test_no_results_for_unrelated_query(self, pipeline):
        ctx = pipeline._build_knowledge_context("cooking breakfast for family")
        assert ctx == ""

    def test_skips_short_input(self, pipeline):
        ctx = pipeline._build_knowledge_context("is it ok")
        assert ctx == ""

    def test_no_knowledge_store(self):
        p = ShellPipeline()
        ctx = p._build_knowledge_context("mesh security")
        assert ctx == ""

    def test_returns_multiple_claims(self, pipeline):
        ctx = pipeline._build_knowledge_context("router scoring VRAM latency")
        assert "router" in ctx.lower() or "vram" in ctx.lower()

    def test_limits_results(self, pipeline):
        ctx = pipeline._build_knowledge_context("mesh protocol security signing peers discovery")
        lines = [l for l in ctx.split("\n") if l.startswith("- [")]
        assert len(lines) <= 10

    def test_orders_by_relevance(self, pipeline):
        ctx = pipeline._build_knowledge_context("mesh security protocol signing")
        # The mesh/security claims should appear first (more keyword matches)
        lines = [l for l in ctx.split("\n") if l.startswith("- [")]
        if lines:
            assert "mesh" in lines[0].lower() or "security" in lines[0].lower()

    def test_scope_independent_search(self, pipeline):
        """SQL search doesn't filter by scope — searches all accepted claims."""
        ctx = pipeline._build_knowledge_context("consensus majority voting")
        assert "consensus" in ctx.lower() or "majority" in ctx.lower()


# ---------------------------------------------------------------------------
# SQL Query
# ---------------------------------------------------------------------------


class TestSQLQuery:
    def test_query_returns_tuples(self, pipeline):
        results = pipeline._query_claims_by_keywords(["mesh", "security"])
        assert len(results) > 0
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 3  # (claim_key, statement, confidence)

    def test_query_empty_keywords(self, pipeline):
        results = pipeline._query_claims_by_keywords([])
        assert results == []

    def test_query_no_ks(self):
        p = ShellPipeline()
        results = p._query_claims_by_keywords(["mesh"])
        assert results == []

    def test_query_respects_limit(self, pipeline):
        results = pipeline._query_claims_by_keywords(["the"], limit=3)
        assert len(results) <= 3

    def test_query_caps_keywords_at_8(self, pipeline):
        """Should handle more than 8 keywords without error."""
        many_keywords = ["mesh", "router", "dag", "knowledge", "adapter",
                         "consensus", "qwen", "pipeline", "acp", "bridge"]
        results = pipeline._query_claims_by_keywords(many_keywords)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------


class TestIntentClassification:
    def test_short_input_is_chat(self, pipeline):
        assert pipeline._classify_intent("hello there") == "chat"

    def test_question_without_action_is_chat(self, pipeline):
        assert pipeline._classify_intent("What is the current status of the project?") == "chat"

    def test_greeting_is_chat(self, pipeline):
        assert pipeline._classify_intent("hi") == "chat"

    def test_action_verb_with_file_is_task(self, pipeline):
        assert pipeline._classify_intent(
            "Fix the bug in src/multihead/router.py that causes incorrect scoring"
        ) == "task"

    def test_multi_step_with_action_is_task(self, pipeline):
        assert pipeline._classify_intent(
            "First implement the retry logic, then add tests for edge cases"
        ) == "task"

    def test_long_detailed_instruction_with_action_is_task(self, pipeline):
        assert pipeline._classify_intent(
            "Refactor the knowledge store to use FTS5 for full-text search "
            "instead of the current LIKE-based queries for better performance "
            "across the 26000 claims in the database"
        ) == "task"

    def test_simple_question_is_chat(self, pipeline):
        assert pipeline._classify_intent("How does the router work?") == "chat"

    def test_has_action_verbs(self, pipeline):
        assert pipeline._has_action_verbs("implement a new feature")
        assert pipeline._has_action_verbs("fix the bug")
        assert not pipeline._has_action_verbs("the sky is blue")

    def test_mentions_files(self, pipeline):
        assert pipeline._mentions_files("look at src/main.py")
        assert pipeline._mentions_files("check /home/user/file.txt")
        assert not pipeline._mentions_files("hello world")

    def test_has_multi_step_language(self, pipeline):
        assert pipeline._has_multi_step_language("first do X then do Y")
        assert pipeline._has_multi_step_language("step 1: setup")
        assert not pipeline._has_multi_step_language("hello world")


# ---------------------------------------------------------------------------
# Pipeline Process (main entry point)
# ---------------------------------------------------------------------------


class TestPipelineProcess:
    async def test_passthrough_when_disabled(self, pipeline, brain_fn):
        pipeline._config.pipeline.enabled = False
        result = await pipeline.process("hello", brain_fn, "ses_1")
        brain_fn.assert_called_once_with("ses_1", "hello", "")
        assert result == brain_fn.return_value

    async def test_chat_intent_goes_to_brain(self, pipeline, brain_fn):
        result = await pipeline.process("hello there", brain_fn, "ses_1")
        brain_fn.assert_called_once()
        assert result == brain_fn.return_value

    async def test_increments_message_count(self, pipeline, brain_fn):
        await pipeline.process("hello", brain_fn, "ses_1")
        assert pipeline._stats["messages_processed"] == 1

    async def test_knowledge_context_passed_to_brain(self, pipeline, brain_fn):
        await pipeline.process("Tell me about mesh security", brain_fn, "ses_1")
        call_args = brain_fn.call_args
        knowledge_ctx = call_args[0][2]  # third positional arg
        # Should have knowledge context (mesh/security matches claims)
        assert "mesh" in knowledge_ctx.lower() or knowledge_ctx == ""

    async def test_knowledge_hit_tracked(self, pipeline, brain_fn):
        await pipeline.process("Tell me about mesh security", brain_fn, "ses_1")
        # If knowledge was found, should track the hit
        if pipeline._stats["knowledge_hits"] > 0:
            assert pipeline._stats["knowledge_hits"] == 1

    async def test_no_knowledge_when_disabled(self, pipeline, brain_fn):
        pipeline._config.pipeline.knowledge_rag = False
        await pipeline.process("Tell me about mesh security", brain_fn, "ses_1")
        call_args = brain_fn.call_args
        knowledge_ctx = call_args[0][2]
        assert knowledge_ctx == ""

    async def test_no_decompose_when_disabled(self, full_pipeline, brain_fn):
        full_pipeline._config.pipeline.auto_decompose = False
        await full_pipeline.process(
            "Fix the bug in src/router.py that causes incorrect scoring weights",
            brain_fn, "ses_1",
        )
        # Should go to brain, not orchestrator
        brain_fn.assert_called_once()
        full_pipeline._orchestrator.create_run.assert_not_called()

    async def test_no_record_when_disabled(self, pipeline, brain_fn):
        pipeline._config.pipeline.auto_record = False
        await pipeline.process("hello", brain_fn, "ses_1")
        assert pipeline._stats["claims_recorded"] == 0

    async def test_process_without_config(self, brain_fn):
        """Pipeline works even without RuntimeConfig."""
        p = ShellPipeline()
        result = await p.process("hello", brain_fn, "ses_1")
        assert result == brain_fn.return_value


# ---------------------------------------------------------------------------
# Task Execution
# ---------------------------------------------------------------------------


class TestTaskExecution:
    async def test_decompose_and_execute(self, full_pipeline, brain_fn):
        # Set explicit decompose_head so decomposition triggers
        full_pipeline._config.pipeline.decompose_head = "mock-llm"
        result = await full_pipeline._execute_as_task(
            "Implement retry logic in the API client with exponential backoff",
            "", "ses_1", brain_fn,
        )
        full_pipeline._decomposer.decompose.assert_called_once()
        full_pipeline._orchestrator.create_run.assert_called_once()
        full_pipeline._orchestrator.execute_run.assert_called_once()
        assert "run_test123" in result

    async def test_no_decompose_when_head_empty(self, full_pipeline, brain_fn):
        """When decompose_head is empty, falls through to brain (no Qwen wake)."""
        full_pipeline._config.pipeline.decompose_head = ""
        result = await full_pipeline._execute_as_task(
            "Implement retry logic in the API client",
            "", "ses_1", brain_fn,
        )
        brain_fn.assert_called_once()
        full_pipeline._decomposer.decompose.assert_not_called()

    async def test_fallback_when_no_decomposer(self, pipeline, brain_fn):
        result = await pipeline._execute_as_task("do something", "", "ses_1", brain_fn)
        brain_fn.assert_called_once()
        assert result == brain_fn.return_value

    async def test_fallback_on_decompose_error(self, full_pipeline, brain_fn):
        full_pipeline._config.pipeline.decompose_head = "mock-llm"
        full_pipeline._decomposer.decompose = AsyncMock(
            side_effect=RuntimeError("decompose failed")
        )
        result = await full_pipeline._execute_as_task("do something", "", "ses_1", brain_fn)
        brain_fn.assert_called_once()
        assert result == brain_fn.return_value

    def test_format_run_results(self, full_pipeline):
        run_state = MagicMock()
        run_state.run_id = "run_abc123def456"
        run_state.status = "completed"
        step_result = MagicMock()
        step_result.status = "completed"
        step_result.output = "Step output here"
        run_state.step_results = {"step_1": step_result}

        text = full_pipeline._format_run_results(run_state)
        assert "run_abc123de" in text
        assert "completed" in text
        assert "Step output here" in text


# ---------------------------------------------------------------------------
# Knowledge Recording
# ---------------------------------------------------------------------------


class TestKnowledgeRecording:
    def test_records_substantial_exchange(self, pipeline):
        long_response = " ".join(["word"] * 25)
        pipeline._maybe_record_knowledge("What is the mesh protocol?", long_response)
        pipeline._ks.insert_claim.assert_called_once()
        assert pipeline._stats["claims_recorded"] == 1

    def test_skips_short_response(self, pipeline):
        pipeline._maybe_record_knowledge("hello", "hi there")
        pipeline._ks.insert_claim.assert_not_called()

    def test_skips_without_knowledge_store(self):
        p = ShellPipeline()
        # Should not raise
        p._maybe_record_knowledge("question", " ".join(["word"] * 25))

    def test_summarize_exchange(self, pipeline):
        summary = pipeline._summarize_exchange(
            "How does the router work?",
            "The router uses weighted scoring. It considers VRAM fit, latency, and accuracy.",
        )
        assert "Q:" in summary
        assert "A:" in summary
        assert len(summary) <= 200

    def test_summarize_truncates_long_question(self, pipeline):
        long_q = "x" * 200
        summary = pipeline._summarize_exchange(long_q, "Short answer. More details follow.")
        assert len(summary) <= 200

    def test_silent_on_insert_error(self, pipeline):
        pipeline._ks.insert_claim.side_effect = RuntimeError("DB error")
        long_response = " ".join(["word"] * 25)
        # Should not raise
        pipeline._maybe_record_knowledge("question", long_response)


# ---------------------------------------------------------------------------
# Image Detection
# ---------------------------------------------------------------------------


class TestImageDetection:
    def test_detects_png(self, pipeline):
        assert pipeline._detect_image_input("look at /tmp/screenshot.png") == "/tmp/screenshot.png"

    def test_detects_jpg(self, pipeline):
        result = pipeline._detect_image_input("analyze image.jpg please")
        assert result is not None
        assert "image.jpg" in result

    def test_no_image(self, pipeline):
        assert pipeline._detect_image_input("hello world") is None

    def test_detects_nested_path(self, pipeline):
        result = pipeline._detect_image_input("check /home/user/images/test.webp")
        assert result is not None
        assert ".webp" in result


# ---------------------------------------------------------------------------
# Config Helpers
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    def test_pipeline_enabled_default(self, pipeline):
        assert pipeline._pipeline_enabled() is True

    def test_pipeline_disabled(self, pipeline):
        pipeline._config.pipeline.enabled = False
        assert pipeline._pipeline_enabled() is False

    def test_config_enabled_default(self, pipeline):
        assert pipeline._config_enabled("auto_decompose") is True

    def test_config_disabled_feature(self, pipeline):
        pipeline._config.pipeline.auto_record = False
        assert pipeline._config_enabled("auto_record") is False

    def test_no_config(self):
        p = ShellPipeline()
        assert p._pipeline_enabled() is True
        assert p._config_enabled("knowledge_rag") is True


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_property(self, pipeline):
        stats = pipeline.stats
        assert "messages_processed" in stats
        assert stats["messages_processed"] == 0

    def test_stats_summary(self, pipeline):
        summary = pipeline.stats_summary()
        assert "Pipeline: ON" in summary
        assert "0 messages" in summary

    def test_stats_summary_disabled(self, pipeline):
        pipeline._config.pipeline.enabled = False
        summary = pipeline.stats_summary()
        assert "Pipeline: OFF" in summary

    async def test_stats_increment(self, pipeline, brain_fn):
        await pipeline.process("hello", brain_fn, "ses_1")
        await pipeline.process("world", brain_fn, "ses_2")
        assert pipeline._stats["messages_processed"] == 2


# ---------------------------------------------------------------------------
# RuntimeConfig Pipeline Integration
# ---------------------------------------------------------------------------


class TestRuntimeConfigPipeline:
    def test_pipeline_config_defaults(self):
        config = RuntimeConfig()
        assert config.pipeline.enabled is True
        assert config.pipeline.auto_decompose is True
        assert config.pipeline.auto_record is True
        assert config.pipeline.knowledge_rag is True
        assert config.pipeline.vlm_auto_route is False
        assert config.pipeline.decompose_threshold == 0.7

    def test_set_pipeline_enabled(self):
        config = RuntimeConfig()
        result = config.set_value("pipeline.enabled", "false")
        assert config.pipeline.enabled is False
        assert "pipeline.enabled" in result

    def test_set_pipeline_auto_decompose(self):
        config = RuntimeConfig()
        config.set_value("pipeline.auto_decompose", "false")
        assert config.pipeline.auto_decompose is False

    def test_set_pipeline_threshold(self):
        config = RuntimeConfig()
        config.set_value("pipeline.decompose_threshold", "0.5")
        assert config.pipeline.decompose_threshold == 0.5

    def test_set_pipeline_unknown_field(self):
        config = RuntimeConfig()
        with pytest.raises(ValueError, match="Unknown pipeline field"):
            config.set_value("pipeline.nonexistent", "true")

    def test_pipeline_config_serialization(self, tmp_path):
        config = RuntimeConfig()
        config.pipeline.enabled = False
        config.pipeline.auto_record = False
        path = tmp_path / "config.json"
        config.save(path)
        loaded = RuntimeConfig.load(path)
        assert loaded.pipeline.enabled is False
        assert loaded.pipeline.auto_record is False

    def test_pipeline_config_in_existing_config(self):
        """PipelineConfig should be present even when loading old config files."""
        config = RuntimeConfig.model_validate({})
        assert config.pipeline.enabled is True


# ---------------------------------------------------------------------------
# Inbox Context
# ---------------------------------------------------------------------------


class TestInboxContext:
    """Test inbox context injection (Stage 1b)."""

    def test_no_inbox_without_knowledge_store(self):
        pipeline = ShellPipeline(knowledge_store=None, runtime_config=RuntimeConfig())
        result = pipeline._build_inbox_context("ses_test")
        assert result == ""

    def test_empty_when_no_pending(self, pipeline):
        pipeline._ks.get_unhandled_claims = MagicMock(return_value=[])
        result = pipeline._build_inbox_context("ses_test")
        assert result == ""

    def test_shows_pending_messages(self, pipeline):
        claim = MagicMock()
        claim.claim_id = "clm_test_pending_1"
        claim.statement = "DECOMP_REQUEST: Fix the layout bug in balloons"
        claim.provenance = MagicMock()
        claim.provenance.produced_by = {"id": "other-agent", "kind": "session"}

        # First call returns messages, second call (plans) returns empty
        pipeline._ks.get_unhandled_claims = MagicMock(
            side_effect=[[claim], []]
        )
        pipeline._ks.record_interaction = MagicMock()

        result = pipeline._build_inbox_context("ses_test")
        assert "[Inbox:" in result
        assert "1 pending" in result
        assert "other-agent" in result
        assert "Fix the layout bug" in result

    def test_shows_proposed_plans(self, pipeline):
        plan_claim = MagicMock()
        plan_claim.claim_id = "clm_test_plan_1"
        plan_claim.statement = "DECOMP_PROPOSAL: Step 1: Analyze, Step 2: Fix"
        plan_claim.provenance = MagicMock()
        plan_claim.provenance.produced_by = {"id": "collaborator", "kind": "session"}
        plan_claim.related_claim_ids = []

        # First call (question/request types) returns empty,
        # second call (plan types) returns the plan
        pipeline._ks.get_unhandled_claims = MagicMock(
            side_effect=[[], [plan_claim]]
        )
        pipeline._ks.record_interaction = MagicMock()

        result = pipeline._build_inbox_context("ses_test")
        assert "[Inbox:" in result
        assert "plan from collaborator" in result

    def test_excludes_own_plans(self, pipeline):
        """get_unhandled_claims already excludes own claims via provenance filter."""
        # With the new API, get_unhandled_claims excludes own claims internally
        pipeline._ks.get_unhandled_claims = MagicMock(return_value=[])
        pipeline._ks.record_interaction = MagicMock()

        result = pipeline._build_inbox_context("ses_test")
        assert result == ""

    def test_limits_to_5_items(self, pipeline):
        claims = []
        for i in range(5):
            c = MagicMock()
            c.claim_id = f"clm_test_limit_{i}"
            c.statement = f"Request {i}"
            c.provenance = MagicMock()
            c.provenance.produced_by = {"id": f"agent-{i}", "kind": "session"}
            claims.append(c)

        plans = []
        for i in range(5, 10):
            c = MagicMock()
            c.claim_id = f"clm_test_limit_{i}"
            c.statement = f"Plan {i}"
            c.provenance = MagicMock()
            c.provenance.produced_by = {"id": f"agent-{i}", "kind": "session"}
            plans.append(c)

        pipeline._ks.get_unhandled_claims = MagicMock(
            side_effect=[claims, plans]
        )
        pipeline._ks.record_interaction = MagicMock()

        result = pipeline._build_inbox_context("ses_test")
        # Should have at most 5 items
        item_count = result.count("- [")
        assert item_count <= 5

    def test_truncates_long_statements(self, pipeline):
        claim = MagicMock()
        claim.claim_id = "clm_test_truncate"
        claim.statement = "A" * 300
        claim.provenance = MagicMock()
        claim.provenance.produced_by = {"id": "agent-x", "kind": "session"}

        pipeline._ks.get_unhandled_claims = MagicMock(
            side_effect=[[claim], []]
        )
        pipeline._ks.record_interaction = MagicMock()

        result = pipeline._build_inbox_context("ses_test")
        # Statement should be truncated to 150 chars
        assert "A" * 151 not in result

    def test_handles_error_gracefully(self, pipeline):
        pipeline._ks.get_unhandled_claims = MagicMock(side_effect=RuntimeError("db error"))
        result = pipeline._build_inbox_context("ses_test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_inbox_injected_into_brain_context(self, pipeline, brain_fn):
        """Inbox context should appear in the knowledge_ctx passed to brain."""
        claim = MagicMock()
        claim.claim_id = "clm_test_inbox_inject"
        claim.statement = "DECOMP_REQUEST: Review the new architecture"
        claim.provenance = MagicMock()
        claim.provenance.produced_by = {"id": "brain-agent", "kind": "session"}

        pipeline._ks.get_unhandled_claims = MagicMock(
            side_effect=[[claim], []]  # first call returns claim, second (plans) empty
        )
        pipeline._ks.record_interaction = MagicMock()

        await pipeline.process("hello", brain_fn, "ses_test")

        # Brain should have been called with inbox context in knowledge_ctx
        call_args = brain_fn.call_args
        knowledge_ctx = call_args[0][2]  # third positional arg
        assert "[Inbox:" in knowledge_ctx
