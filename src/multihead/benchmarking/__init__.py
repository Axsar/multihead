"""Benchmarking engine for solver evaluation.

Runs standardized benchmarks to measure solver performance:
- LLM: MMLU, GSM8K, HumanEval
- Vision: COCO mAP, latency on test images
- Deterministic: Correctness tests
"""

from multihead.benchmarking.base import Benchmark, BenchmarkResult, BenchmarkRunner
from multihead.benchmarking.llm_benchmarks import MMLUBenchmark, GSM8KBenchmark, SimpleReasoningBenchmark
from multihead.benchmarking.vision_benchmarks import LatencyBenchmark, ImageClassificationBenchmark, COCOBenchmark
from multihead.benchmarking.code_benchmarks import HumanEvalBenchmark

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkRunner",
    "MMLUBenchmark",
    "GSM8KBenchmark",
    "SimpleReasoningBenchmark",
    "LatencyBenchmark",
    "ImageClassificationBenchmark",
    "COCOBenchmark",
    "HumanEvalBenchmark",
]
