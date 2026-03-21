"""Claim fusion — cross-reference claims from independent observation channels.

Implements the triangulation pattern:
1. Group claims by shared anchors (file_path, symbol_name)
2. For each group, compare claims from different observation_methods
3. Score convergence/divergence on a gradient
4. Update claim confidence and lifecycle status

Three states per claim-channel pair:
- CONFIRMED: another channel agrees
- CONTRADICTED: another channel disagrees
- SILENT: other channels have nothing to say

Principles:
- Independence is sacred — channels must observe independently
- Divergence = signal, not noise — surface it, don't resolve it
- Silence != confirmation — absence of evidence is not evidence
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of fusing claims about a single topic."""
    anchor: str  # The file_path or symbol that groups these claims
    claims: list[dict]  # [{claim_id, statement, observation_method, confidence}]
    channel_states: dict[str, str]  # {observation_method: CONFIRMED|CONTRADICTED|SILENT}
    convergence_score: float  # -1.0 (hard diverge) to +1.0 (full converge)
    confidence_adjustments: dict[str, float]  # {claim_id: adjustment}
    action: str  # "boost" | "penalize" | "flag_contradiction" | "no_change"
    # LLM judgment categories for divergent pairs: {claim_id: category}
    # Categories: TEMPORAL_DRIFT, CONFIG_DRIFT, IMPROVEMENT_UNVERIFIED, CONTRADICT
    divergence_categories: dict[str, str] = field(default_factory=dict)


@dataclass
class FusionReport:
    """Summary of a full fusion run."""
    topics_analyzed: int = 0
    claims_boosted: int = 0
    claims_penalized: int = 0
    contradictions_found: int = 0
    corroborated: int = 0
    unverified: int = 0
    temporal_drift: int = 0
    config_drift: int = 0
    improvement_unverified: int = 0


class ClaimFusion:
    """Cross-reference claims from different observation channels.

    Pattern:
    1. Gather all claims about a topic (grouped by anchor)
    2. Group by observation_method
    3. Score pairwise convergence
    4. Update confidence and lifecycle
    """

    # Known project roots — absolute paths that should be stripped for matching.
    # Configured via MULTIHEAD_PROJECT_ROOTS env var (colon-separated).
    _PROJECT_ROOTS: tuple[str, ...] = ()  # populated dynamically

    def __init__(self, knowledge_store, generate_fn=None):
        self.ks = knowledge_store
        # Sync function: (prompt: str) -> dict with "text" key
        # Caller is responsible for bridging async→sync if needed
        self._generate_fn = generate_fn

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize file paths for consistent grouping.

        Strips known project root prefixes so absolute and relative paths
        match: '/workspace/myproject/src/router.py' → 'src/router.py'.
        """
        if not path:
            return ""
        from multihead._paths import normalize_file_path
        result = normalize_file_path(path)
        if result != path:
            return result
        # Also handle generic /mnt/*/DevD/*/ patterns
        import re
        m = re.match(r'/mnt/[a-z]/[^/]+/[^/]+/', path)
        if m:
            return path[len(m.group(0)):]
        return path

    def run_fusion(self, scope_id: str | None = None, limit: int = 100000) -> FusionReport:
        """Run fusion across all anchored claims.

        Groups claims by file_path anchor, compares across observation methods,
        updates confidence and lifecycle status.
        """
        report = FusionReport()
        now = datetime.now(timezone.utc).isoformat()

        # Get all claims with source_anchor.file_path
        anchored = self._get_anchored_claims(scope_id, limit)
        if not anchored:
            return report

        # Group by file_path (normalized to relative paths for consistent matching)
        by_anchor: dict[str, list[dict]] = {}
        for claim in anchored:
            anchor = self._normalize_path(claim.get("file_path", ""))
            if anchor:
                by_anchor.setdefault(anchor, []).append(claim)

        for anchor, claims in by_anchor.items():
            # Need at least 2 claims to fuse
            if len(claims) < 2:
                report.unverified += 1
                continue

            result = self._fuse_topic(anchor, claims)
            report.topics_analyzed += 1

            # Apply results to DB
            self._apply_fusion_result(result, now)

            # Count divergence categories
            for cat in result.divergence_categories.values():
                if cat == "TEMPORAL_DRIFT" or cat == "TEMPORAL":
                    report.temporal_drift += 1
                elif cat == "CONFIG_DRIFT":
                    report.config_drift += 1
                elif cat == "IMPROVEMENT_UNVERIFIED":
                    report.improvement_unverified += 1

            if result.action == "boost":
                report.claims_boosted += len([v for v in result.confidence_adjustments.values() if v > 0])
                report.corroborated += 1
            elif result.action == "penalize":
                report.claims_penalized += len([v for v in result.confidence_adjustments.values() if v < 0])
            elif result.action == "flag_contradiction":
                report.contradictions_found += 1

        logger.info(
            "Fusion complete: %d topics, %d boosted, %d penalized, %d contradictions, "
            "%d unverified | categories: %d temporal, %d config, %d unverified_improvement",
            report.topics_analyzed, report.claims_boosted, report.claims_penalized,
            report.contradictions_found, report.unverified,
            report.temporal_drift, report.config_drift, report.improvement_unverified,
        )
        return report

    def _fuse_topic(self, anchor: str, claims: list[dict]) -> FusionResult:
        """Fuse claims about a single topic (file/symbol).

        Groups by observation_method, then scores pairwise convergence.
        """
        # Group by INDEPENDENT CHANNEL, not raw observation_method.
        # user_statement + assistant_statement from conversations = ONE channel.
        # Only truly independent sources count as separate channels.
        CONVERSATION_METHODS = {"user_statement", "assistant_statement", "conversation_analysis", "conversation", ""}
        by_method: dict[str, list[dict]] = {}
        for c in claims:
            method = c.get("observation_method", "unknown")
            # Merge conversation-derived methods into one channel
            if method in CONVERSATION_METHODS:
                channel = "conversation"
            else:
                channel = method  # code_read, code_behavior_llm, git_history, bash_output, document_read — each independent
            by_method.setdefault(channel, []).append(c)

        methods = list(by_method.keys())
        channel_states: dict[str, str] = {}
        adjustments: dict[str, float] = {}

        # Single method — UNVERIFIED
        if len(methods) <= 1:
            for c in claims:
                channel_states[c.get("observation_method", "unknown")] = "SILENT"
            return FusionResult(
                anchor=anchor,
                claims=claims,
                channel_states=channel_states,
                convergence_score=0.0,
                confidence_adjustments={},
                action="no_change",
            )

        # Multiple methods — score pairwise convergence
        convergence_scores: list[float] = []
        div_categories: dict[str, str] = {}  # claim_id → category

        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                claims_a = by_method[method_a]
                claims_b = by_method[method_b]

                # Score convergence between the two groups
                score, category = self._score_convergence(claims_a, claims_b)
                convergence_scores.append(score)

                # Track divergence categories per claim
                if category and score < -0.1:
                    for c in claims_a + claims_b:
                        div_categories[c["claim_id"]] = category

                if score > 0.7:
                    # Full convergence
                    channel_states[method_a] = "CONFIRMED"
                    channel_states[method_b] = "CONFIRMED"
                    for c in claims_a + claims_b:
                        adjustments[c["claim_id"]] = 0.15
                elif score > 0.3:
                    # Compatible
                    channel_states.setdefault(method_a, "CONFIRMED")
                    channel_states.setdefault(method_b, "CONFIRMED")
                    for c in claims_a + claims_b:
                        adjustments.setdefault(c["claim_id"], 0.05)
                elif score > -0.3:
                    # Neutral — silence
                    channel_states.setdefault(method_a, "SILENT")
                    channel_states.setdefault(method_b, "SILENT")
                elif score > -0.7:
                    # Soft divergence
                    channel_states[method_a] = "CONTRADICTED"
                    channel_states[method_b] = "CONTRADICTED"
                    for c in claims_a + claims_b:
                        adjustments[c["claim_id"]] = -0.2
                else:
                    # Hard divergence
                    channel_states[method_a] = "CONTRADICTED"
                    channel_states[method_b] = "CONTRADICTED"
                    for c in claims_a + claims_b:
                        adjustments[c["claim_id"]] = -0.3

        avg_convergence = sum(convergence_scores) / len(convergence_scores) if convergence_scores else 0.0

        if avg_convergence > 0.3:
            action = "boost"
        elif avg_convergence < -0.3:
            action = "flag_contradiction"
        elif any(v < -0.1 for v in adjustments.values()):
            action = "penalize"
        else:
            action = "no_change"

        return FusionResult(
            anchor=anchor,
            claims=claims,
            channel_states=channel_states,
            convergence_score=avg_convergence,
            confidence_adjustments=adjustments,
            action=action,
            divergence_categories=div_categories,
        )

    def _score_convergence(self, claims_a: list[dict], claims_b: list[dict]) -> tuple[float, str]:
        """Score convergence between two sets of claims from different channels.

        Returns (score, category) where:
        - score: -1.0 (hard diverge) to +1.0 (full converge)
        - category: SAME_FACT, AGREE, TEMPORAL_DRIFT, CONFIG_DRIFT,
                     IMPROVEMENT_UNVERIFIED, CONTRADICT, NEUTRAL, or ""
        """
        text_a = " ".join(c.get("statement", "") for c in claims_a)
        text_b = " ".join(c.get("statement", "") for c in claims_b)

        if not text_a.strip() or not text_b.strip():
            return 0.0, ""  # Neutral — one side has nothing

        # Try embedding similarity first (best quality)
        similarity = self._embedding_similarity(text_a, text_b)

        # Check for explicit contradiction signals
        lower_a, lower_b = text_a.lower(), text_b.lower()
        contra_signals = [
            ("not ", "is "), ("was ", "now "), ("broken", "working"),
            ("removed", "added"), ("disabled", "enabled"),
            ("deprecated", "active"), ("no longer", "currently"),
            ("failed", "passed"), ("missing", "present"),
        ]
        contradiction_count = sum(
            1 for neg, pos in contra_signals
            if (neg in lower_a and pos in lower_b) or (pos in lower_a and neg in lower_b)
        )

        # Detect change-claim patterns for CONFIG_DRIFT / IMPROVEMENT_UNVERIFIED
        methods_a = {c.get("observation_method", "") for c in claims_a}
        methods_b = {c.get("observation_method", "") for c in claims_b}
        has_code_read = bool({"code_read", "code_behavior_llm"} & (methods_a | methods_b))
        has_conversation = bool({"conversation", "user_statement", "assistant_statement"} & (methods_a | methods_b))

        # Check for "removed/added/replaced/improved/fixed" change claims
        change_words = {"removed", "added", "replaced", "improved", "fixed", "changed", "updated", "refactored"}
        text_has_change = bool(change_words & set(lower_a.split())) or bool(change_words & set(lower_b.split()))

        # Compute convergence score
        if contradiction_count >= 2:
            cat = self._classify_divergence(
                has_code_read, has_conversation, text_has_change, lower_a, lower_b,
            )
            return -0.8, cat
        elif contradiction_count == 1:
            cat = self._classify_divergence(
                has_code_read, has_conversation, text_has_change, lower_a, lower_b,
            )
            return min(-0.3, similarity - 0.5), cat
        elif similarity > 0.9:
            return min(1.0, similarity), "SAME_FACT"
        elif similarity > 0.5:
            # AMBIGUOUS ZONE — send to LLM for judgment
            llm_score, llm_cat = self._llm_judge_convergence(text_a, text_b)
            if llm_score is not None:
                # Refine category based on channel context
                if llm_cat in ("TEMPORAL", "INTENT_VS_IMPL", "CONTRADICT") and has_code_read and has_conversation:
                    refined = self._classify_divergence(
                        has_code_read, has_conversation, text_has_change, lower_a, lower_b,
                    )
                    return llm_score, refined
                return llm_score, llm_cat
            return similarity * 0.6, ""
        elif similarity > 0.2:
            return 0.0, "NEUTRAL"
        else:
            return 0.0, ""

    def _llm_judge_convergence(self, text_a: str, text_b: str) -> tuple[float | None, str]:
        """Ask an LLM whether two claims agree or contradict.

        Returns (score, category) where score is -1.0 to +1.0 and category
        is the raw LLM judgment (SAME_FACT, AGREE, TEMPORAL, etc.).
        Raises RuntimeError if no generate_fn was provided.
        """
        if self._generate_fn is None:
            raise RuntimeError(
                "Fusion LLM tier requires a generate function. "
                "Pass generate_fn to ClaimFusion or run fusion stage with an LLM adapter. "
                "Without LLM judgment, fusion produces 96% false positives."
            )

        prompt = (
            "Two claims about the same code/system:\n\n"
            f"Claim A: {text_a}\n\n"
            f"Claim B: {text_b}\n\n"
            "Classify their relationship:\n"
            "- SAME_FACT: Same information stated in different words\n"
            "- AGREE: Different but compatible facts that support each other\n"
            "- TEMPORAL: One describes old state, other describes current state (before/after a change)\n"
            "- INTENT_VS_IMPL: One describes what SHOULD be done (decision/intent), other describes what IS done (implementation)\n"
            "- PARTIAL: Related but different aspects of the same topic\n"
            "- CONTRADICT: They state incompatible facts (different numbers, different behaviors for the same thing)\n\n"
            "Reply with ONLY one word: SAME_FACT, AGREE, TEMPORAL, INTENT_VS_IMPL, PARTIAL, or CONTRADICT"
        )

        try:
            # generate_fn is sync — caller bridges async if needed
            result = self._generate_fn(prompt)
            answer = result.get("text", "").strip().upper()

            if "CONTRADICT" in answer:
                return -0.7, "CONTRADICT"
            elif "TEMPORAL" in answer:
                return -0.4, "TEMPORAL"
            elif "INTENT_VS_IMPL" in answer:
                return -0.2, "INTENT_VS_IMPL"
            elif "SAME_FACT" in answer:
                return 0.9, "SAME_FACT"
            elif "AGREE" in answer:
                return 0.8, "AGREE"
            elif "PARTIAL" in answer:
                return 0.3, "PARTIAL"

            logger.warning("LLM judge returned unrecognized answer: %s", answer)
            return None, ""

        except Exception as e:
            logger.error("LLM judge failed: %s", e)
            raise  # Don't swallow — silent failure is unacceptable

    @staticmethod
    def _classify_divergence(
        has_code_read: bool,
        has_conversation: bool,
        text_has_change: bool,
        lower_a: str,
        lower_b: str,
    ) -> str:
        """Classify divergence into sub-category based on channel and content context.

        Used by both keyword heuristic and LLM tiers to ensure consistent categorization.
        """
        if not has_code_read and not has_conversation:
            return "CONTRADICT"  # Can't classify without recognizable channel context

        # CONFIG_DRIFT: "removed X" / "set to 0" but code still has it
        config_signals = {"removed", "set to 0", "disabled", "default", "no longer"}
        if any(s in lower_a or s in lower_b for s in config_signals):
            return "CONFIG_DRIFT"

        # IMPROVEMENT_UNVERIFIED: "improved" / "fixed" / "replaced" but code_read disagrees
        improve_signals = {"improved", "fixed", "replaced", "better", "upgraded", "refactored"}
        if any(s in lower_a or s in lower_b for s in improve_signals):
            return "IMPROVEMENT_UNVERIFIED"

        # TEMPORAL_DRIFT: intent/past-state that may not match current code.
        # Requires a change signal to be present — avoids matching meta-claims that
        # merely discuss temporal drift as a concept (e.g. claim_key contains "temporal").
        temporal_signals = {"decided to", "we will", "was changed", "used to", "previously", "at the time"}
        if text_has_change and any(s in lower_a or s in lower_b for s in temporal_signals):
            return "TEMPORAL_DRIFT"

        # Generic change words
        if text_has_change:
            return "IMPROVEMENT_UNVERIFIED"

        return "CONTRADICT"

    def _embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts using embeddings.

        Falls back to Jaccard word overlap if embeddings unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            if not hasattr(self, "_model"):
                self._model = SentenceTransformer("all-MiniLM-L6-v2")

            emb_a = self._model.encode(text_a, normalize_embeddings=True)
            emb_b = self._model.encode(text_b, normalize_embeddings=True)
            return float(np.dot(emb_a, emb_b))

        except ImportError:
            # Fallback: word overlap
            words_a = set(re.findall(r'[a-z_][a-z0-9_]+', text_a.lower()))
            words_b = set(re.findall(r'[a-z_][a-z0-9_]+', text_b.lower()))
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

    def _apply_fusion_result(self, result: FusionResult, now: str) -> None:
        """Apply fusion confidence adjustments and lifecycle updates to the DB."""
        with self.ks._connect() as conn:
            for claim in result.claims:
                cid = claim["claim_id"]
                adj = result.confidence_adjustments.get(cid, 0.0)

                if adj != 0.0:
                    # Update confidence
                    new_conf = max(0.1, min(0.95, claim.get("confidence", 0.5) + adj))
                    conn.execute(
                        "UPDATE claims SET confidence = ?, updated_at = ? WHERE claim_id = ?",
                        (new_conf, now, cid),
                    )

                # Lifecycle based on fusion outcome
                state = result.channel_states.get(claim.get("observation_method", ""), "SILENT")
                # Map convergence score to lifecycle action
                score = result.convergence_score
                if state == "CONFIRMED":
                    conn.execute(
                        "UPDATE claims SET claim_status = 'corroborated', updated_at = ? "
                        "WHERE claim_id = ? AND claim_status = 'proposed'",
                        (now, cid),
                    )
                # Check per-claim divergence category (independent of average score)
                cat = result.divergence_categories.get(cid, "")
                obs = claim.get("observation_method", "")
                is_conversation = obs in ("conversation", "user_statement", "assistant_statement")

                if cat and cat != "NEUTRAL":

                    if cat in ("TEMPORAL_DRIFT", "TEMPORAL"):
                        # R&D churn — conversation claims about changed code → STALE
                        conn.execute(
                            "UPDATE claims SET claim_status = 'stale', "
                            "contested_reason = ?, updated_at = ? "
                            "WHERE claim_id = ? AND claim_status IN ('proposed', 'corroborated')",
                            (f"[TEMPORAL_DRIFT] {score:.2f}", now, cid),
                        )
                    elif cat == "CONFIG_DRIFT":
                        # Config/flag drift — code_read is ground truth for structural facts
                        if obs in ("code_read", "code_behavior_llm"):
                            # Code observation wins — keep it
                            pass
                        else:
                            conn.execute(
                                "UPDATE claims SET claim_status = 'contested', "
                                "contested_reason = ?, updated_at = ? "
                                "WHERE claim_id = ?",
                                (f"[CONFIG_DRIFT] code_read disagrees: {score:.2f}", now, cid),
                            )
                    elif cat == "IMPROVEMENT_UNVERIFIED":
                        # Claimed improvement not confirmed by code_read
                        # Mark conversation claims as contested (code_read is ground truth)
                        if is_conversation:
                            conn.execute(
                                "UPDATE claims SET claim_status = 'contested', "
                                "contested_reason = ?, updated_at = ? "
                                "WHERE claim_id = ?",
                                (f"[IMPROVEMENT_UNVERIFIED] code_read does not confirm: {score:.2f}", now, cid),
                            )
                    elif score <= -0.5:
                        # Hard divergence / genuine contradiction
                        conn.execute(
                            "UPDATE claims SET claim_status = 'contested', "
                            "contested_reason = ?, updated_at = ? "
                            "WHERE claim_id = ? AND claim_status IN ('proposed', 'corroborated')",
                            (f"[CONTRADICT] {score:.2f}", now, cid),
                        )
                    elif is_conversation and score <= -0.3:
                        # Soft divergence on conversation claims → STALE
                        conn.execute(
                            "UPDATE claims SET claim_status = 'stale', "
                            "contested_reason = ?, updated_at = ? "
                            "WHERE claim_id = ? AND claim_status IN ('proposed', 'corroborated')",
                            (f"[TEMPORAL_DRIFT] soft divergence: {score:.2f}", now, cid),
                        )

    def _get_anchored_claims(self, scope_id: str | None, limit: int) -> list[dict]:
        """Get claims that have file_path in their source_anchor."""
        with self.ks._connect() as conn:
            where = "provenance_json LIKE '%file_path%' AND claim_status NOT IN ('superseded', 'rejected')"
            params: list = []
            if scope_id:
                where += " AND scope_id = ?"
                params.append(scope_id)
            params.append(limit)

            rows = conn.execute(
                f"SELECT claim_id, statement, confidence, provenance_json, claim_status "
                f"FROM claims WHERE {where} LIMIT ?",
                params,
            ).fetchall()

        if len(rows) >= limit:
            logger.warning(
                "Fusion hit claim limit (%d) — some anchored claims excluded. "
                "Raise limit or filter by scope to cover all claims.",
                limit,
            )
        if len(rows) > 10000:
            logger.warning("Fusion processing large anchored claim set: %d claims", len(rows))

        claims = []
        for row in rows:
            try:
                prov = json.loads(row["provenance_json"])
                anchor = prov.get("source_anchor", {})
                claims.append({
                    "claim_id": row["claim_id"],
                    "statement": row["statement"],
                    "confidence": row["confidence"],
                    "observation_method": prov.get("observation_method", "unknown"),
                    "file_path": anchor.get("file_path", ""),
                    "symbol": anchor.get("symbol", ""),
                    "claim_status": row["claim_status"],
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return claims
