"""Deterministic adapter for zero-cost, perfect-accuracy solvers.

Executes pure Python functions without LLM inference.
Benefits:
- Perfect accuracy (no hallucination)
- Zero cost (no API calls, no GPU)
- Instant execution (<1ms typically)
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from ..deterministic import DETERMINISTIC_FUNCTIONS, get_function


class DeterministicAdapter:
    """Adapter for deterministic (non-ML) solver functions."""

    def __init__(self, model: str):
        """Initialize deterministic adapter.

        Args:
            model: Function reference in format:
                   - "function_name" (looks up in DETERMINISTIC_FUNCTIONS)
                   - "module.function" (imports from module)

        Example:
            >>> adapter = DeterministicAdapter("coordinate_transform")
            >>> adapter = DeterministicAdapter("multihead.deterministic.coordinate_transform")
        """
        self.model = model
        self.function = self._load_function(model)

    def _load_function(self, model: str):
        """Load function from registry or import from module."""
        # Try built-in registry first
        if model in DETERMINISTIC_FUNCTIONS:
            return DETERMINISTIC_FUNCTIONS[model]

        # Try module import
        if "." in model:
            module_path, function_name = model.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                return getattr(module, function_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"Could not import deterministic function '{model}': {e}"
                )

        # Not found
        available = ", ".join(DETERMINISTIC_FUNCTIONS.keys())
        raise ValueError(
            f"Deterministic function '{model}' not found. "
            f"Available built-ins: {available}"
        )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,  # Ignored (deterministic)
        max_tokens: int = 1000,  # Ignored
        **kwargs
    ) -> dict[str, Any]:
        """Execute deterministic function and return result dict.

        Args:
            prompt: JSON string with function arguments
            temperature: Ignored (deterministic functions always deterministic)
            max_tokens: Ignored
            **kwargs: Passed to function

        Returns:
            Dict with 'text' (JSON string), 'tokens_in', 'tokens_out'

        Example:
            >>> prompt = json.dumps({
            ...     "points": [(100, 200), (300, 400)],
            ...     "source_resolution": [3975, 6150],
            ...     "target_resolution": [4042, 5929]
            ... })
            >>> result = await adapter.generate(prompt)
            >>> result
            {'text': '{"transformed_points": [[101.68, 193.14], ...]}', 'tokens_in': 0, 'tokens_out': 0}
        """
        # Parse prompt as JSON (should contain function arguments)
        try:
            args = json.loads(prompt)
        except json.JSONDecodeError:
            # If prompt is not JSON, assume it's kwargs dict
            args = {"input": prompt}

        # Handle different input formats
        if isinstance(args, dict):
            # Dict of kwargs
            function_args = args
        elif isinstance(args, list):
            # List of positional args (convert to first kwarg)
            function_args = {"data": args}
        else:
            # Single value
            function_args = {"input": args}

        # Merge with adapter kwargs
        function_args.update(kwargs)

        # Execute function
        try:
            result = self.function(**function_args)

            # Ensure result is serializable
            if not isinstance(result, dict):
                result = {"output": result}

            # Return in standard adapter format (same as MockAdapter, OpenAIAdapter, etc.)
            return {
                "text": json.dumps(result),
                "tokens_in": 0,  # Deterministic = zero cost
                "tokens_out": 0,
            }

        except Exception as e:
            # Return error as JSON
            error_result = {
                "error": str(e),
                "function": self.model,
                "args": function_args
            }
            return {
                "text": json.dumps(error_result),
                "tokens_in": 0,
                "tokens_out": 0,
            }

    async def load(self):
        """No loading needed for deterministic functions (instant startup)."""
        pass

    async def unload(self):
        """No unloading needed for deterministic functions."""
        pass

    async def sleep(self, level: int = 1):
        """No sleep needed for deterministic functions (no memory footprint)."""
        pass

    async def wake(self):
        """No wake needed for deterministic functions (always ready)."""
        pass

    async def generate_stream(self, prompt: str, **kwargs):
        """Deterministic functions return instantly, no streaming."""
        result = await self.generate(prompt, **kwargs)
        yield result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Not supported for deterministic adapter."""
        raise NotImplementedError(
            "Deterministic adapter does not support embeddings"
        )

    async def cleanup(self):
        """No cleanup needed for deterministic functions."""
        pass

    def __repr__(self):
        return f"DeterministicAdapter(model={self.model})"
