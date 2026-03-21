"""Built-in validators for common contract patterns.

Provides concrete Validator implementations:
- JSONSchemaValidator: JSON schema validation
- FormatValidator: Length, regex, enum constraints
- ConfidenceValidator: Confidence threshold checks
- ContractValidator: Arbitrary Python predicates
- CompositeValidator: AND/OR composition of validators
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal

from .base import ValidationResult, Validator


# ---------------------------------------------------------------------------
# JSON Schema Validator
# ---------------------------------------------------------------------------

class JSONSchemaValidator(Validator):
    """Validates JSON output against a schema using jsonschema library.

    Example:
        >>> validator = JSONSchemaValidator(schema={
        ...     "type": "object",
        ...     "required": ["verdict", "confidence"],
        ...     "properties": {
        ...         "verdict": {"type": "string", "enum": ["approve", "reject"]},
        ...         "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        ...     }
        ... })
        >>> result = validator.validate_postcondition({}, {"verdict": "approve", "confidence": 0.9})
        >>> assert result.passed
    """

    name = "json_schema"

    def __init__(
        self,
        schema: dict[str, Any],
        require_precondition_schema: bool = False
    ):
        """
        Args:
            schema: JSON schema dict (follows jsonschema specification)
            require_precondition_schema: If True, validate inputs against schema too
        """
        self.schema = schema
        self.require_precondition_schema = require_precondition_schema

    def validate_precondition(
        self,
        state: dict[str, Any],
        inputs: dict[str, Any]
    ) -> ValidationResult:
        if not self.require_precondition_schema:
            return ValidationResult(passed=True)

        # Validate inputs match expected schema (for JSON -> JSON transforms)
        input_data = inputs.get("input_data", {})
        return self._validate_json(input_data)

    def validate_postcondition(
        self,
        state: dict[str, Any],
        output: Any
    ) -> ValidationResult:
        # Parse output if string
        data = self._parse_output(output)
        return self._validate_json(data)

    def _parse_output(self, output: Any) -> Any:
        """Parse output to dict if JSON string."""
        if isinstance(output, str):
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                # Return as-is, validation will fail if schema expects object
                return output
        return output

    def _validate_json(self, data: Any) -> ValidationResult:
        """Validate data against schema."""
        try:
            from jsonschema import validate, ValidationError

            validate(instance=data, schema=self.schema)
            return ValidationResult(passed=True, confidence=1.0)

        except ValidationError as e:
            return ValidationResult(
                passed=False,
                confidence=1.0,  # We're certain it failed
                violations=[f"Schema violation: {e.message}"],
                metadata={
                    "path": list(e.path),
                    "validator": e.validator,
                    "validator_value": e.validator_value
                }
            )
        except ImportError:
            # jsonschema not installed
            return ValidationResult(
                passed=False,
                confidence=0.0,
                violations=["jsonschema library not installed"],
                metadata={"install": "pip install jsonschema"}
            )


# ---------------------------------------------------------------------------
# Format Validator
# ---------------------------------------------------------------------------

class FormatValidator(Validator):
    """Validates output format constraints (length, regex, enum).

    Example:
        >>> validator = FormatValidator(
        ...     min_length=10,
        ...     max_length=750,  # MAKER red-flag threshold
        ...     regex=r"^\\{.*\\}$",  # Must be JSON object
        ...     allowed_values=None
        ... )
        >>> result = validator.validate_postcondition({}, '{"key": "value"}')
        >>> assert result.passed
    """

    name = "format"

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        regex: str | None = None,
        allowed_values: list[str] | None = None,
        check_precondition: bool = False
    ):
        """
        Args:
            min_length: Minimum allowed length (characters)
            max_length: Maximum allowed length (MAKER uses 750 as red-flag threshold)
            regex: Regex pattern output must match
            allowed_values: List of allowed string values (enum)
            check_precondition: If True, also validate input format
        """
        self.min_length = min_length
        self.max_length = max_length
        self.regex = regex
        self.allowed_values = allowed_values
        self.check_precondition = check_precondition

    def validate_precondition(
        self,
        state: dict[str, Any],
        inputs: dict[str, Any]
    ) -> ValidationResult:
        if not self.check_precondition:
            return ValidationResult(passed=True)

        prompt = inputs.get("prompt", "")
        return self._validate_format(str(prompt))

    def validate_postcondition(
        self,
        state: dict[str, Any],
        output: Any
    ) -> ValidationResult:
        return self._validate_format(str(output))

    def _validate_format(self, text: str) -> ValidationResult:
        """Validate text against format constraints."""
        violations = []

        if self.min_length is not None and len(text) < self.min_length:
            violations.append(
                f"Length {len(text)} below minimum {self.min_length}"
            )

        if self.max_length is not None and len(text) > self.max_length:
            violations.append(
                f"Length {len(text)} exceeds maximum {self.max_length}"
            )

        if self.regex is not None:
            if not re.search(self.regex, text, re.DOTALL):
                violations.append(f"Does not match regex: {self.regex}")

        if self.allowed_values is not None:
            if text not in self.allowed_values:
                violations.append(
                    f"Value '{text[:100]}...' not in allowed: {self.allowed_values}"
                )

        return ValidationResult(
            passed=len(violations) == 0,
            confidence=1.0,
            violations=violations,
            metadata={
                "actual_length": len(text),
                "min_length": self.min_length,
                "max_length": self.max_length
            }
        )


# ---------------------------------------------------------------------------
# Confidence Validator
# ---------------------------------------------------------------------------

class ConfidenceValidator(Validator):
    """Validates confidence/quality thresholds in output.

    Expects output to contain a confidence score (0.0-1.0) in a specific field.
    Useful for filtering low-confidence LLM outputs or triggering refinement loops.

    Example:
        >>> validator = ConfidenceValidator(min_confidence=0.7)
        >>> result = validator.validate_postcondition(
        ...     {},
        ...     {"verdict": "approve", "confidence": 0.85}
        ... )
        >>> assert result.passed
        >>> assert result.confidence == 0.85  # Uses output's own confidence
    """

    name = "confidence"

    def __init__(
        self,
        min_confidence: float = 0.7,
        confidence_field: str = "confidence",
        extract_from_json: bool = True
    ):
        """
        Args:
            min_confidence: Minimum acceptable confidence score (0.0-1.0)
            confidence_field: Field name in output dict containing score
            extract_from_json: If True, parse JSON strings to extract score
        """
        self.min_confidence = min_confidence
        self.confidence_field = confidence_field
        self.extract_from_json = extract_from_json

    def validate_precondition(
        self,
        state: dict[str, Any],
        inputs: dict[str, Any]
    ) -> ValidationResult:
        # Confidence validation only makes sense for postconditions
        return ValidationResult(passed=True)

    def validate_postcondition(
        self,
        state: dict[str, Any],
        output: Any
    ) -> ValidationResult:
        confidence = self._extract_confidence(output)

        if confidence is None:
            return ValidationResult(
                passed=False,
                confidence=0.5,  # Uncertain about failure
                violations=[
                    f"Could not extract confidence from field '{self.confidence_field}'"
                ],
                metadata={
                    "output_type": type(output).__name__,
                    "expected_field": self.confidence_field
                }
            )

        passed = confidence >= self.min_confidence

        return ValidationResult(
            passed=passed,
            confidence=confidence,  # Use output's own confidence as ours
            violations=[] if passed else [
                f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}"
            ],
            metadata={
                "actual_confidence": confidence,
                "threshold": self.min_confidence
            }
        )

    def _extract_confidence(self, output: Any) -> float | None:
        """Extract confidence score from output."""
        # Try dict access
        if isinstance(output, dict):
            val = output.get(self.confidence_field)
            if isinstance(val, (int, float)):
                return float(val)

        # Try JSON parsing
        if self.extract_from_json and isinstance(output, str):
            try:
                data = json.loads(output)
                if isinstance(data, dict):
                    val = data.get(self.confidence_field)
                    if isinstance(val, (int, float)):
                        return float(val)
            except json.JSONDecodeError:
                pass

        return None


# ---------------------------------------------------------------------------
# Contract Validator
# ---------------------------------------------------------------------------

class ContractValidator(Validator):
    """Validates arbitrary Python predicates for custom contracts.

    Use this for domain-specific invariants or complex relationships
    that don't fit into schema/format validators.

    Example:
        >>> def check_coordinate_space(state, output):
        ...     if isinstance(output, dict):
        ...         space = output.get("coordinate_space")
        ...         valid = space in ["page_resolution", "unet_fixed"]
        ...         return valid, f"coordinate_space is {space}"
        ...     return False, "Output not a dict"
        >>>
        >>> validator = ContractValidator(
        ...     postcondition=check_coordinate_space,
        ...     name="coordinate_space_contract"
        ... )
    """

    name = "contract"

    def __init__(
        self,
        precondition: Callable[[dict, dict], tuple[bool, str]] | None = None,
        postcondition: Callable[[dict, Any], tuple[bool, str]] | None = None,
        contract_name: str = "custom_contract"
    ):
        """
        Args:
            precondition: (state, inputs) -> (passed: bool, reason: str)
            postcondition: (state, output) -> (passed: bool, reason: str)
            contract_name: Name for debugging/logging
        """
        self.precondition_fn = precondition
        self.postcondition_fn = postcondition
        self.contract_name = contract_name

    def validate_precondition(
        self,
        state: dict[str, Any],
        inputs: dict[str, Any]
    ) -> ValidationResult:
        if not self.precondition_fn:
            return ValidationResult(passed=True)

        try:
            passed, reason = self.precondition_fn(state, inputs)
            return ValidationResult(
                passed=passed,
                confidence=1.0,
                violations=[] if passed else [f"{self.contract_name}: {reason}"],
                metadata={"contract": self.contract_name}
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                confidence=0.5,
                violations=[f"Precondition check raised exception: {e}"],
                metadata={"exception": str(e), "contract": self.contract_name}
            )

    def validate_postcondition(
        self,
        state: dict[str, Any],
        output: Any
    ) -> ValidationResult:
        if not self.postcondition_fn:
            return ValidationResult(passed=True)

        try:
            passed, reason = self.postcondition_fn(state, output)
            return ValidationResult(
                passed=passed,
                confidence=1.0,
                violations=[] if passed else [f"{self.contract_name}: {reason}"],
                metadata={"contract": self.contract_name}
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                confidence=0.5,
                violations=[f"Postcondition check raised exception: {e}"],
                metadata={"exception": str(e), "contract": self.contract_name}
            )


# ---------------------------------------------------------------------------
# Composite Validator
# ---------------------------------------------------------------------------

class CompositeValidator(Validator):
    """Combines multiple validators with AND/OR logic.

    Use this to enforce multiple constraints on a single step
    (e.g., schema + length + confidence all must pass).

    Example:
        >>> validator = CompositeValidator(
        ...     validators=[
        ...         JSONSchemaValidator(schema={"type": "object"}),
        ...         FormatValidator(max_length=750),  # MAKER red-flag
        ...         ConfidenceValidator(min_confidence=0.7)
        ...     ],
        ...     mode="all"  # All three must pass
        ... )
    """

    name = "composite"

    def __init__(
        self,
        validators: list[Validator],
        mode: Literal["all", "any"] = "all"
    ):
        """
        Args:
            validators: List of validators to combine
            mode: "all" (AND) - all must pass
                  "any" (OR) - at least one must pass
        """
        self.validators = validators
        self.mode = mode

    def validate_precondition(
        self,
        state: dict[str, Any],
        inputs: dict[str, Any]
    ) -> ValidationResult:
        results = [
            v.validate_precondition(state, inputs)
            for v in self.validators
        ]
        return self._combine_results(results)

    def validate_postcondition(
        self,
        state: dict[str, Any],
        output: Any
    ) -> ValidationResult:
        results = [
            v.validate_postcondition(state, output)
            for v in self.validators
        ]
        return self._combine_results(results)

    def _combine_results(
        self,
        results: list[ValidationResult]
    ) -> ValidationResult:
        """Combine multiple validation results with AND/OR logic."""
        if self.mode == "all":
            # All must pass (AND)
            passed = all(r.passed for r in results)
            confidence = min(r.confidence for r in results)  # Weakest link
            violations = [
                v
                for r in results
                for v in r.violations
            ]
        else:
            # Any must pass (OR)
            passed = any(r.passed for r in results)
            confidence = max(r.confidence for r in results)  # Strongest
            # Only report violations from failed validators
            violations = [
                v
                for r in results
                if not r.passed
                for v in r.violations
            ]

        return ValidationResult(
            passed=passed,
            confidence=confidence,
            violations=violations,
            metadata={
                "mode": self.mode,
                "validator_count": len(results),
                "validators": [v.name for v in self.validators]
            }
        )
