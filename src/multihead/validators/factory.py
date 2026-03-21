"""Validator factory for declarative (YAML) configuration.

Provides create_validator_from_config() to instantiate validators
from plain dicts, enabling YAML recipes to specify validators
without writing Python code.
"""

from __future__ import annotations

from typing import Any

from .base import Validator
from .builtin import (
    CompositeValidator,
    ConfidenceValidator,
    FormatValidator,
    JSONSchemaValidator,
)


def create_validator_from_config(config: dict[str, Any]) -> Validator:
    """Create a validator instance from a configuration dict.

    This enables YAML recipes to specify validators declaratively.

    Example:
        >>> config = {
        ...     "type": "composite",
        ...     "mode": "all",
        ...     "validators": [
        ...         {"type": "json_schema", "schema": {"type": "object"}},
        ...         {"type": "format", "max_length": 750}
        ...     ]
        ... }
        >>> validator = create_validator_from_config(config)

    Args:
        config: Dict with "type" key and validator-specific params

    Returns:
        Validator instance

    Raises:
        ValueError: If validator type unknown
    """
    validator_type = config.get("type")

    if validator_type == "json_schema":
        return JSONSchemaValidator(
            schema=config.get("schema", {}),
            require_precondition_schema=config.get("require_precondition_schema", False)
        )

    elif validator_type == "format":
        return FormatValidator(
            min_length=config.get("min_length"),
            max_length=config.get("max_length"),
            regex=config.get("regex"),
            allowed_values=config.get("allowed_values"),
            check_precondition=config.get("check_precondition", False)
        )

    elif validator_type == "confidence":
        return ConfidenceValidator(
            min_confidence=config.get("min_confidence", 0.7),
            confidence_field=config.get("confidence_field", "confidence"),
            extract_from_json=config.get("extract_from_json", True)
        )

    elif validator_type == "contract":
        # Contract validators can't be created from YAML (need Python functions)
        raise ValueError(
            "ContractValidator requires Python functions, cannot create from YAML. "
            "Use ContractValidator(...) directly in Python code."
        )

    elif validator_type == "composite":
        sub_validators = [
            create_validator_from_config(v)
            for v in config.get("validators", [])
        ]
        return CompositeValidator(
            validators=sub_validators,
            mode=config.get("mode", "all")
        )

    else:
        raise ValueError(
            f"Unknown validator type: {validator_type}. "
            f"Supported: json_schema, format, confidence, composite"
        )
