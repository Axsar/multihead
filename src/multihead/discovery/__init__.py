"""Solver discovery system for MultiHead.

Automatically discovers new models and solvers from:
- HuggingFace Model Hub
- Ollama library
- BotVibes marketplace
- Papers with Code

Benchmarks and updates the solver registry.
"""

from multihead.discovery.base import DiscoveryAgent, SolverCandidate
from multihead.discovery.huggingface import HuggingFaceDiscovery
from multihead.discovery.ollama import OllamaDiscovery
from multihead.discovery.botvibes import BotVibesDiscovery
from multihead.discovery.paperswithcode import PapersWithCodeDiscovery

__all__ = [
    "DiscoveryAgent",
    "SolverCandidate",
    "HuggingFaceDiscovery",
    "OllamaDiscovery",
    "BotVibesDiscovery",
    "PapersWithCodeDiscovery",
]
