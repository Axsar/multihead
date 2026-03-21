"""Consensus strategy implementations, schema validation, and red-flag detection.

Provides :class:`ConsensusStrategiesMixin` which is mixed into
:class:`~multihead.consensus.engine.ConsensusEngine` to keep the engine
module focused on orchestration.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from .models import (
    ConsensusConfig,
    ConsensusStrategy,
    FirstToAheadConfig,
    HeadTask,
    VoteResult,
)


class ConsensusStrategiesMixin:
    """Mixin supplying voting strategies, validation, and helpers."""

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _apply_strategy(
        self,
        votes: list[VoteResult],
        config: ConsensusConfig,
    ) -> tuple[dict[str, Any], float]:
        """Apply consensus strategy. Returns (outputs, agreement_score)."""
        match config.strategy:
            case ConsensusStrategy.MAJORITY:
                return self._majority_vote(votes)
            case ConsensusStrategy.WEIGHTED:
                head_map = {ht.head_id: ht for ht in config.heads}
                return self._weighted_vote(votes, head_map)
            case ConsensusStrategy.UNANIMOUS:
                return self._unanimous_check(votes)
            case ConsensusStrategy.THRESHOLD:
                return self._threshold_vote(votes, config.threshold)
            case _:
                return self._majority_vote(votes)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _majority_vote(
        self, votes: list[VoteResult],
    ) -> tuple[dict[str, Any], float]:
        """Simple majority vote on text output."""
        texts = [v.outputs.get("text", "") for v in votes]
        counter = Counter(texts)

        if not counter:
            return {}, 0.0

        winner, count = counter.most_common(1)[0]
        agreement = count / len(texts)

        # Use the full outputs from the first vote that matches the winner
        consensus_outputs = {"text": winner}
        for v in votes:
            if v.outputs.get("text", "") == winner:
                consensus_outputs = dict(v.outputs)
                break

        return consensus_outputs, agreement

    def _weighted_vote(
        self,
        votes: list[VoteResult],
        head_map: dict[str, HeadTask],
    ) -> tuple[dict[str, Any], float]:
        """Weighted vote: higher-weight heads have more influence."""
        weighted_counts: dict[str, float] = {}
        total_weight = 0.0

        for v in votes:
            text = v.outputs.get("text", "")
            weight = head_map.get(v.head_id, HeadTask(head_id=v.head_id)).weight
            weighted_counts[text] = weighted_counts.get(text, 0.0) + weight
            total_weight += weight

        if not weighted_counts or total_weight == 0:
            return {}, 0.0

        winner = max(weighted_counts, key=weighted_counts.get)  # type: ignore[arg-type]
        agreement = weighted_counts[winner] / total_weight

        consensus_outputs = {"text": winner}
        for v in votes:
            if v.outputs.get("text", "") == winner:
                consensus_outputs = dict(v.outputs)
                break

        return consensus_outputs, agreement

    def _unanimous_check(
        self, votes: list[VoteResult],
    ) -> tuple[dict[str, Any], float]:
        """All votes must produce identical text output."""
        texts = [v.outputs.get("text", "") for v in votes]

        if not texts:
            return {}, 0.0

        all_same = all(t == texts[0] for t in texts)

        if all_same:
            return dict(votes[0].outputs), 1.0

        # Not unanimous -- return the majority as best guess, score = fraction agreeing
        counter = Counter(texts)
        winner, count = counter.most_common(1)[0]
        consensus_outputs = {"text": winner}
        for v in votes:
            if v.outputs.get("text", "") == winner:
                consensus_outputs = dict(v.outputs)
                break

        return consensus_outputs, count / len(texts)

    def _threshold_vote(
        self,
        votes: list[VoteResult],
        threshold: float,
    ) -> tuple[dict[str, Any], float]:
        """Majority vote with minimum agreement threshold."""
        consensus_outputs, agreement = self._majority_vote(votes)

        # If below threshold, still return outputs but with the low score
        # Red flag detection will handle the threshold breach
        return consensus_outputs, agreement

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def _validate_schema(
        self, text: str, schema: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate output text against a JSON schema (basic checks)."""
        errors: list[str] = []

        expected_type = schema.get("type")
        if not expected_type:
            return True, []

        if expected_type == "object":
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    errors.append(f"Expected object, got {type(data).__name__}")
                else:
                    # Check required fields
                    required = schema.get("required", [])
                    for field in required:
                        if field not in data:
                            errors.append(f"Missing required field: {field}")
            except json.JSONDecodeError:
                errors.append("Output is not valid JSON")

        elif expected_type == "array":
            try:
                data = json.loads(text)
                if not isinstance(data, list):
                    errors.append(f"Expected array, got {type(data).__name__}")
            except json.JSONDecodeError:
                errors.append("Output is not valid JSON")

        elif expected_type == "string":
            if not isinstance(text, str):
                errors.append(f"Expected string, got {type(text).__name__}")

        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Cross-modal verification
    # ------------------------------------------------------------------

    def _verify_cross_modal(
        self,
        votes: list[VoteResult],
        head_configs: list[HeadTask],
    ) -> list[dict[str, Any]]:
        """Verify consistency across different modalities.

        For each head with extract_fields, parse its output and collect
        the field values. Then check that overlapping fields agree.
        """
        flags: list[dict[str, Any]] = []
        head_config_map = {ht.head_id: ht for ht in head_configs}

        # Collect extracted fields from each head
        field_values: dict[str, list[tuple[str, Any]]] = {}  # field -> [(head_id, value)]

        for vote in votes:
            config = head_config_map.get(vote.head_id)
            if not config or not config.extract_fields:
                continue

            # Try to parse output as JSON for field extraction
            text = vote.outputs.get("text", "")
            parsed = self._try_parse_json(text)

            for field_name in config.extract_fields:
                value = None
                if parsed is not None and isinstance(parsed, dict):
                    value = parsed.get(field_name)
                elif parsed is not None:
                    value = parsed
                else:
                    # Use raw text for non-JSON outputs
                    value = text

                if field_name not in field_values:
                    field_values[field_name] = []
                field_values[field_name].append((vote.head_id, value))

        # Check for conflicts: if same field extracted by multiple heads, they should agree
        for field_name, entries in field_values.items():
            if len(entries) < 2:
                continue

            values = [v for _, v in entries]
            heads_involved = [h for h, _ in entries]

            # Normalize for comparison
            normalized = [self._normalize_value(v) for v in values]
            unique_values = set(normalized)

            if len(unique_values) > 1:
                flags.append({
                    "type": "cross_modal_conflict",
                    "severity": "high",
                    "message": (
                        f"Field '{field_name}' disagrees across heads: "
                        f"{', '.join(f'{h}={v!r}' for h, v in entries)}"
                    ),
                    "details": {
                        "field": field_name,
                        "heads": heads_involved,
                        "values": values,
                    },
                })

        return flags

    # ------------------------------------------------------------------
    # Red flag detection
    # ------------------------------------------------------------------

    def _detect_red_flags(
        self,
        all_votes: list[VoteResult],
        valid_votes: list[VoteResult],
        consensus_outputs: dict[str, Any],
        config: ConsensusConfig,
    ) -> list[dict[str, Any]]:
        """Detect disagreements, failures, and schema violations."""
        flags: list[dict[str, Any]] = []
        head_config_map = {ht.head_id: ht for ht in config.heads}

        # 1. Required head failures
        for vote in all_votes:
            if not vote.success:
                ht = head_config_map.get(vote.head_id, HeadTask(head_id=vote.head_id))
                severity = "critical" if ht.required else "medium"
                flags.append({
                    "type": "head_failure",
                    "severity": severity,
                    "message": f"Head '{vote.head_id}' failed: {vote.error}",
                    "details": {"head_id": vote.head_id, "required": ht.required},
                })

        # 2. Schema violations
        for vote in all_votes:
            if vote.success and not vote.schema_valid:
                flags.append({
                    "type": "schema_violation",
                    "severity": "medium",
                    "message": (
                        f"Head '{vote.head_id}' output failed schema validation: "
                        f"{'; '.join(vote.schema_errors)}"
                    ),
                    "details": {"head_id": vote.head_id, "errors": vote.schema_errors},
                })

        # 3. Low agreement (for threshold strategy)
        if config.strategy == ConsensusStrategy.THRESHOLD and valid_votes:
            texts = [v.outputs.get("text", "") for v in valid_votes]
            counter = Counter(texts)
            if counter:
                _, count = counter.most_common(1)[0]
                agreement = count / len(texts)
                if agreement < config.threshold:
                    flags.append({
                        "type": "low_agreement",
                        "severity": "high",
                        "message": (
                            f"Agreement {agreement:.0%} below threshold {config.threshold:.0%}"
                        ),
                        "details": {"agreement": agreement, "threshold": config.threshold},
                    })

        # 4. No valid votes at all
        if not valid_votes:
            flags.append({
                "type": "no_valid_votes",
                "severity": "critical",
                "message": "No heads produced valid output",
                "details": {"total_heads": len(all_votes)},
            })

        return flags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        """Try to parse text as JSON, return None on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _normalize_value(value: Any) -> str:
        """Normalize a value for comparison (JSON-serialize)."""
        if value is None:
            return "null"
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value).strip().lower()

    @staticmethod
    def _canonical_hash(text: str) -> str:
        """Canonical hash for equivalence bucketing.

        Try parse as JSON -> sort keys -> compact dump -> SHA256[:16].
        If not JSON: strip + lowercase -> SHA256[:16].
        """
        try:
            parsed = json.loads(text)
            normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _check_red_flags_pre_vote(
        vote: VoteResult,
        policy: FirstToAheadConfig,
        output_schema: dict[str, Any],
    ) -> tuple[bool, str]:
        """Pre-vote red-flag filter (MAKER's key reliability win).

        Returns (is_flagged, reason). Flagged samples are discarded, not repaired.
        """
        if not vote.success:
            return True, f"head_error: {vote.error}"

        text = vote.outputs.get("text", "")

        # Token-length heuristic (words / 0.75 ~ tokens)
        approx_tokens = len(text.split()) / 0.75
        if approx_tokens > policy.red_flag_max_tokens:
            return True, f"too_long: ~{int(approx_tokens)} tokens > {policy.red_flag_max_tokens}"

        # JSON parse requirement
        if policy.red_flag_must_parse or output_schema.get("type") in ("object", "array"):
            try:
                parsed = json.loads(text)
                # Check required fields if schema specifies them
                required = output_schema.get("required", [])
                if isinstance(parsed, dict):
                    missing = [f for f in required if f not in parsed]
                    if missing:
                        return True, f"missing_fields: {missing}"
            except (json.JSONDecodeError, TypeError):
                return True, "json_parse_failure"

        return False, ""
