"""Validation for decomposed task plans.

Provides atomicity validation (MAKER m=1 verification) and
completeness validation (goal coverage checking).
"""

from __future__ import annotations

from multihead.decomposer import DecompositionPlan, TaskNode


# ---------------------------------------------------------------------------
# Atomic step validation
# ---------------------------------------------------------------------------


class AtomicityValidator:
    """Validates that steps are truly atomic (m=1 from MAKER).

    A step is atomic if it performs exactly ONE concrete action:
    - Single file read/write
    - Single test execution
    - Single verification check

    Non-atomic examples:
    - "Read file X and file Y" -> Should be 2 steps
    - "Edit function A and function B" -> Should be 2 steps
    - "Run tests and verify results" -> Should be 2 steps
    """

    # Words indicating multiple actions
    MULTI_ACTION_WORDS = frozenset({"and", "&", "then", "also", "additionally", "plus"})

    # Words indicating single actions
    SINGLE_ACTION_VERBS = frozenset({
        "read", "write", "edit", "create", "delete", "test", "verify",
        "check", "run", "execute", "call", "invoke", "load", "save",
    })

    # Verb combinations that are still considered atomic (synonyms or elaborations)
    ATOMIC_VERB_COMBINATIONS = [
        {"read", "understand"},  # read to understand
        {"read", "review"},  # read and review
        {"read", "test"},  # read test cases
        {"run", "test"},  # run tests
        {"run", "execute"},  # run/execute (synonyms)
        {"edit", "save"},  # edit and save
        {"create", "save"},  # create and save
    ]

    # Common atomic phrases that might trigger false positives
    # These are complete phrases, not substring matches
    ATOMIC_PHRASES = [
        "read and understand",
        "read to understand",
        "read test cases",
        "read test case",
        "run tests",  # But not "edit then run tests"
    ]

    @classmethod
    def is_atomic(cls, leaf: TaskNode) -> tuple[bool, str | None]:
        """Check if a leaf step is atomic.

        Args:
            leaf: Leaf TaskNode to validate

        Returns:
            (is_atomic, reason) tuple. reason is None if atomic, else describes why not.
        """
        goal = leaf.goal.lower()

        # Check 0: Whitelist common atomic phrases first
        # Only whitelist if the phrase IS the goal (not just contained in it)
        goal_clean = goal.strip()
        for phrase in cls.ATOMIC_PHRASES:
            # Allow exact match or match with minor additions like "the"
            if goal_clean == phrase or goal_clean.startswith(phrase + " the") or goal_clean.startswith("the " + phrase):
                return True, None

        # Check 1: Multiple targets
        if len(leaf.target_files) > 1:
            return False, f"Targets {len(leaf.target_files)} files (should be 1)"

        # Check 2: Multiple action words (and/then/also)
        word_count = sum(1 for word in cls.MULTI_ACTION_WORDS if f" {word} " in f" {goal} ")
        if word_count > 0:
            return False, f"Contains multi-action words: {', '.join(w for w in cls.MULTI_ACTION_WORDS if f' {w} ' in f' {goal} ')}"

        # Check 3: Multiple verb phrases
        verbs_found = [v for v in cls.SINGLE_ACTION_VERBS if v in goal]
        if len(verbs_found) > 1:
            # Check if this is an allowed combination (still atomic)
            verbs_set = set(verbs_found)
            for allowed_combo in cls.ATOMIC_VERB_COMBINATIONS:
                if verbs_set == allowed_combo or verbs_set.issubset(allowed_combo):
                    return True, None
            return False, f"Contains multiple actions: {', '.join(verbs_found)}"

        # Check 4: Goal length heuristic (>100 chars likely composite)
        if len(leaf.goal) > 100:
            return False, f"Goal too long ({len(leaf.goal)} chars, likely composite)"

        return True, None

    @classmethod
    def validate_plan(cls, plan: DecompositionPlan) -> list[tuple[str, str]]:
        """Validate all leaves in a plan for atomicity.

        Args:
            plan: DecompositionPlan to validate

        Returns:
            List of (step_id, reason) tuples for non-atomic steps
        """
        violations = []
        for leaf in plan.all_leaves():
            is_atomic, reason = cls.is_atomic(leaf)
            if not is_atomic:
                violations.append((leaf.id, reason or "Unknown"))
        return violations


# ---------------------------------------------------------------------------
# Completeness validation
# ---------------------------------------------------------------------------


class CompletenessValidator:
    """Validates that decomposition covers the goal.

    Checks:
    - Goal keywords appear in step descriptions
    - Critical action types present (explore -> understand -> implement -> verify)
    - No orphan steps (disconnected from goal)
    """

    STANDARD_PHASES = {
        "explore": ["explore", "read", "investigate"],
        "understand": ["read", "analyze", "review"],
        "implement": ["edit", "create", "refactor"],
        "verify": ["test", "verify", "check"],
    }

    @classmethod
    def validate(cls, goal: str, plan: DecompositionPlan) -> tuple[bool, list[str]]:
        """Validate plan completeness.

        Args:
            goal: Original goal text
            plan: DecompositionPlan to validate

        Returns:
            (is_complete, issues) tuple. is_complete=True if no critical issues.
        """
        issues = []

        # Check 1: Goal keywords coverage
        goal_keywords = cls._extract_keywords(goal)
        plan_text = " ".join(leaf.goal.lower() for leaf in plan.all_leaves())

        missing_keywords = []
        for kw in goal_keywords:
            if kw not in plan_text:
                missing_keywords.append(kw)

        if missing_keywords:
            issues.append(f"Goal keywords not in plan: {', '.join(missing_keywords[:3])}")

        # Check 2: Standard phase coverage
        action_types = {leaf.action_type for leaf in plan.all_leaves()}

        # For complex tasks, expect explore -> implement -> verify
        if plan.complexity in ("moderate", "complex"):
            if not any(a in action_types for a in cls.STANDARD_PHASES["explore"]):
                issues.append("Missing exploration phase (no read/investigate steps)")
            if not any(a in action_types for a in cls.STANDARD_PHASES["implement"]):
                issues.append("Missing implementation phase (no edit/create steps)")
            if not any(a in action_types for a in cls.STANDARD_PHASES["verify"]):
                issues.append("Missing verification phase (no test/verify steps)")

        # Check 3: Empty plan
        if plan.total_steps == 0:
            issues.append("Plan has no steps")

        # Check 4: Single-step plan for complex goal
        if plan.total_steps == 1 and plan.complexity == "complex":
            issues.append("Only 1 step for complex goal (likely under-decomposed)")

        is_complete = len(issues) == 0
        return is_complete, issues

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract important keywords from text."""
        from multihead.decomposer import _extract_keywords
        return _extract_keywords(text)
