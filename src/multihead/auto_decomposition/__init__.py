"""Auto-Decomposition — Enhanced LLM-driven task breakdown with DAG inference.

Extends TaskDecomposer with:
- DAG dependency inference (parallel vs sequential steps)
- Atomic step validation (m=1 verification)
- Integration with ToT/PRM/Reflection
- Completeness validation

Based on research findings:
- MAKER: Maximal Agentic Decomposition (m=1)
- K-Step Reasoning: Atomic steps with verification
- ADaPT: As-Needed Decomposition based on complexity
"""

from .decomposer import AutoDecomposer
from .dependency import StepDependencyAnalyzer
from .research import ResearchFeatureIntegrator
from .validators import AtomicityValidator, CompletenessValidator

__all__ = [
    "AutoDecomposer",
    "AtomicityValidator",
    "CompletenessValidator",
    "ResearchFeatureIntegrator",
    "StepDependencyAnalyzer",
]
