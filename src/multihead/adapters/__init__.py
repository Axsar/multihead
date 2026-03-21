"""Head adapters for different model backends."""

from .anthropic_adapter import AnthropicAdapter
from .base import HeadAdapter
from .botvibes_adapter import BotVibesAdapter
from .claude_adapter import ClaudeAdapter
from .claude_agent_sdk import ClaudeAgentSDKAdapter
from .claude_session import ClaudeSessionAdapter
from .embedding import EmbeddingAdapter
from .mock import MockAdapter
from .ollama import OllamaAdapter
from .openai_adapter import OpenAIAdapter
from .transformers_adapter import TransformersAdapter
from .vllm import VLLMAdapter

__all__ = [
    "AnthropicAdapter",
    "HeadAdapter",
    "BotVibesAdapter",
    "ClaudeAdapter",
    "ClaudeAgentSDKAdapter",
    "ClaudeSessionAdapter",
    "EmbeddingAdapter",
    "MockAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "TransformersAdapter",
    "VLLMAdapter",
]
