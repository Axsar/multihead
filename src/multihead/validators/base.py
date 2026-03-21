"""Core types for contract-based validators.

Provides the ValidationResult model and Validator ABC that all
concrete validators inherit from.

See docs/design-atomic-step-contracts.md for design details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Validation Result
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of a contract validation check.

    Attributes:
        passed: Whether validation passed (True) or failed (False)
        confidence: Confidence in validation result (0.0-1.0)
                   1.0 = certain (schema validation, exact match)
                   0.5-0.99 = uncertain (LLM-based, heuristic)
                   <0.5 = low confidence (flag for human review)
        violations: Human-readable descriptions of constraint violations
        metadata: Machine-readable context for debugging
    """

    passed: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    violations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base Validator
# ---------------------------------------------------------------------------

class Validator(ABC):
    """Base class for step contract validators.

    Validators implement Hoare-style contracts:
    - validate_precondition: Check if state allows execution
    - validate_postcondition: Check if output meets guarantees

    Validators can be composed via CompositeValidator for complex constraints.
    """

    name: str = "base_validator"

    @abstractmethod
    def validate_precondition(
        self,
        state: dict[str, Any],
        inputs: dict[str, Any]
    ) -> ValidationResult:
        """Check if preconditions are met before step execution.

        Args:
            state: Current run state (step_results, artifacts, etc.)
            inputs: Step inputs being passed (from input_refs)

        Returns:
            ValidationResult indicating if precondition is satisfied

        Note:
            If precondition fails, step execution is skipped entirely.
        """
        pass

    @abstractmethod
    def validate_postcondition(
        self,
        state: dict[str, Any],
        output: Any
    ) -> ValidationResult:
        """Check if postconditions are met after step execution.

        Args:
            state: Updated run state after execution
            output: Step output (str, dict, etc.)

        Returns:
            ValidationResult indicating if output meets guarantees

        Note:
            If postcondition fails, output is discarded and step may retry.
        """
        pass
