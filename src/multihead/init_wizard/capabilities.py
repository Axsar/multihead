"""Model capability database and inference logic."""

from __future__ import annotations

from typing import Any


# --- Capability inference database for known models ---
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    # Local LLMs
    "qwen3:4b": {
        "solver_type": "llm", "kind": "llm",
        "input_modalities": ["text"], "output_modalities": ["text", "json"],
        "task_types": ["semantic_reasoning", "text_generation", "classification"],
        "max_input_tokens": 32768, "latency_p50_ms": 80, "cost_per_call": 0.0,
        "accuracy_score": 0.62,
        "benchmarks": {"mmlu": 0.62, "gsm8k": 0.72},
    },
    "qwen3:8b": {
        "solver_type": "llm", "kind": "llm",
        "input_modalities": ["text"], "output_modalities": ["text", "json"],
        "task_types": ["semantic_reasoning", "text_generation", "classification",
                       "summarization", "question_answering"],
        "max_input_tokens": 32768, "latency_p50_ms": 120, "cost_per_call": 0.0,
        "accuracy_score": 0.71,
        "benchmarks": {"mmlu": 0.71, "gsm8k": 0.83, "humaneval": 0.68},
    },
    "qwen3:32b": {
        "solver_type": "llm", "kind": "llm",
        "input_modalities": ["text"], "output_modalities": ["text", "json"],
        "task_types": ["semantic_reasoning", "text_generation", "classification",
                       "summarization", "code_generation", "question_answering"],
        "max_input_tokens": 32768, "latency_p50_ms": 300, "cost_per_call": 0.0,
        "accuracy_score": 0.82,
        "benchmarks": {"mmlu": 0.82, "gsm8k": 0.91, "humaneval": 0.78},
    },
    # Transformers models
    "Qwen/Qwen3-8B": {
        "solver_type": "llm", "kind": "llm",
        "input_modalities": ["text"], "output_modalities": ["text", "json"],
        "task_types": ["semantic_reasoning", "text_generation", "classification",
                       "summarization", "question_answering"],
        "max_input_tokens": 32768, "latency_p50_ms": 120, "cost_per_call": 0.0,
        "accuracy_score": 0.71,
        "benchmarks": {"mmlu": 0.71, "gsm8k": 0.83, "humaneval": 0.68},
    },
    "Qwen/Qwen3-VL-7B": {
        "solver_type": "vlm", "kind": "vlm",
        "input_modalities": ["text", "image"], "output_modalities": ["text", "json"],
        "task_types": ["visual_reasoning", "image_classification", "image_description",
                       "visual_question_answering", "ocr"],
        "max_input_tokens": 8192, "latency_p50_ms": 150, "cost_per_call": 0.0,
        "accuracy_score": 0.68,
        "benchmarks": {"mmmu": 0.68, "docvqa": 0.82},
    },
    "Qwen/Qwen3-VL-32B-Thinking": {
        "solver_type": "vlm", "kind": "vlm",
        "input_modalities": ["text", "image"], "output_modalities": ["text", "json"],
        "task_types": ["visual_reasoning", "complex_image_analysis", "image_classification",
                       "image_description", "visual_question_answering", "ocr",
                       "spatial_reasoning"],
        "max_input_tokens": 32768, "latency_p50_ms": 400, "cost_per_call": 0.0,
        "accuracy_score": 0.94,
        "benchmarks": {"mmmu": 0.78, "docvqa": 0.91},
    },
    # Cloud models
    "sonnet": {
        "solver_type": "llm", "kind": "llm",
        "input_modalities": ["text", "image"], "output_modalities": ["text", "json", "code"],
        "task_types": ["semantic_reasoning", "text_generation", "classification",
                       "summarization", "code_generation", "question_answering",
                       "analysis", "creative_writing"],
        "max_input_tokens": 200000, "latency_p50_ms": 1200, "cost_per_call": 0.003,
        "accuracy_score": 0.92,
        "benchmarks": {"mmlu": 0.92, "gsm8k": 0.96, "humaneval": 0.92},
    },
    "claude-opus-4-6": {
        "solver_type": "llm", "kind": "llm",
        "input_modalities": ["text", "image"], "output_modalities": ["text", "json", "code"],
        "task_types": ["semantic_reasoning", "text_generation", "code_generation",
                       "question_answering", "analysis", "creative_writing",
                       "agentic_execution"],
        "max_input_tokens": 200000, "latency_p50_ms": 2000, "cost_per_call": 0.005,
        "accuracy_score": 0.95,
        "benchmarks": {"mmlu": 0.92, "gsm8k": 0.96, "humaneval": 0.92},
    },
    "gpt-4o-mini": {
        "solver_type": "llm", "kind": "llm",
        "input_modalities": ["text", "image"], "output_modalities": ["text", "json"],
        "task_types": ["semantic_reasoning", "text_generation", "classification",
                       "summarization", "code_generation", "question_answering"],
        "max_input_tokens": 128000, "latency_p50_ms": 800, "cost_per_call": 0.0025,
        "accuracy_score": 0.88,
        "benchmarks": {"mmlu": 0.88, "gsm8k": 0.95, "humaneval": 0.87},
    },
}

DEFAULT_LLM_CAPABILITIES: dict[str, Any] = {
    "solver_type": "llm",
    "input_modalities": ["text"],
    "output_modalities": ["text", "json"],
    "task_types": ["semantic_reasoning", "text_generation"],
    "latency_p50_ms": 200, "cost_per_call": 0.0,
    "accuracy_score": 0.5,
}

DEFAULT_VLM_CAPABILITIES: dict[str, Any] = {
    "solver_type": "vlm",
    "input_modalities": ["text", "image"],
    "output_modalities": ["text", "json"],
    "task_types": ["visual_reasoning", "image_classification"],
    "latency_p50_ms": 300, "cost_per_call": 0.0,
    "accuracy_score": 0.5,
}


def infer_capabilities(head: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Infer capability metadata for a head based on model name and adapter.

    Returns:
        Tuple of (capabilities_dict, privacy_level).
    """
    model = head.get("model", "")
    kind = head.get("kind", "llm")
    adapter = head.get("adapter", "")

    # Try exact model match first
    if model in MODEL_CAPABILITIES:
        caps = dict(MODEL_CAPABILITIES[model])
    else:
        # Fallback by kind
        if kind == "vlm":
            caps = dict(DEFAULT_VLM_CAPABILITIES)
        else:
            caps = dict(DEFAULT_LLM_CAPABILITIES)

    # Enrich with head-specific data
    gpu = head.get("gpu_required", False)
    caps["requires_gpu"] = gpu
    caps["vram_mb"] = head.get("vram_hint_mb", 0)
    caps["cost_per_call"] = caps.get("cost_per_call", 0.0)

    # Adapter-based privacy inference
    if adapter in ("mock", "transformers", "ollama"):
        privacy = "local"
    elif adapter in ("claude", "claude_agent_sdk", "openai"):
        privacy = "encrypted"
    elif adapter == "botvibes":
        privacy = "encrypted"
    else:
        privacy = "local"

    # Remove kind from caps (it's a head-level field, not capability)
    caps.pop("kind", None)

    return caps, privacy
