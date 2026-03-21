"""Quality Verifier: Sample-based quality checking for delegated tasks."""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

from .models import QualityVerificationReport

logger = logging.getLogger(__name__)


class QualityVerifier:
    """Verifies quality of marketplace results via sampling."""

    def __init__(
        self,
        sample_rate: float = 0.05,  # 5% sampling by default
        min_samples: int = 5,
        max_samples: int = 100,
    ):
        """Initialize quality verifier.

        Args:
            sample_rate: Fraction of results to sample (0.0-1.0)
            min_samples: Minimum samples even for small datasets
            max_samples: Maximum samples even for large datasets
        """
        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.max_samples = max_samples

    def verify_results(
        self,
        rfq_id: str,
        provider_id: str,
        results: list[Any],
        ground_truth: list[Any] | None = None,
        quality_fn: Callable[[Any, Any], float] | None = None,
        threshold: float = 0.85,
    ) -> QualityVerificationReport:
        """Verify quality of marketplace results.

        Args:
            rfq_id: RFQ identifier
            provider_id: Provider who produced results
            results: Results to verify
            ground_truth: Optional ground truth for comparison
            quality_fn: Function to compute quality score (result, truth) -> score
            threshold: Minimum acceptable quality

        Returns:
            Quality verification report
        """
        total_results = len(results)

        # Determine sample size
        sample_size = max(
            self.min_samples,
            min(self.max_samples, int(total_results * self.sample_rate))
        )
        sample_size = min(sample_size, total_results)

        # Sample random indices
        sample_indices = sorted(random.sample(range(total_results), sample_size))

        # Verify samples
        quality_scores = []
        passed_count = 0
        failed_count = 0
        failure_reasons = []

        for idx in sample_indices:
            result = results[idx]

            # Compute quality score
            if ground_truth and quality_fn:
                truth = ground_truth[idx] if idx < len(ground_truth) else None
                if truth is None:
                    logger.warning("No ground truth for sample %d", idx)
                    failure_reasons.append(f"Sample {idx}: no ground truth")
                    failed_count += 1
                    continue

                try:
                    score = quality_fn(result, truth)
                    quality_scores.append(score)

                    if score >= threshold:
                        passed_count += 1
                    else:
                        failed_count += 1
                        failure_reasons.append(
                            f"Sample {idx}: score {score:.3f} < threshold {threshold}"
                        )
                except Exception as e:
                    logger.error("Quality check failed for sample %d: %s", idx, e)
                    failed_count += 1
                    failure_reasons.append(f"Sample {idx}: error - {str(e)}")
            else:
                # No ground truth or quality function
                # Just verify result is not None/empty
                if result is not None and result != "":
                    passed_count += 1
                    quality_scores.append(1.0)
                else:
                    failed_count += 1
                    failure_reasons.append(f"Sample {idx}: empty result")

        # Compute average quality
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        meets_threshold = average_quality >= threshold

        report = QualityVerificationReport(
            rfq_id=rfq_id,
            provider_id=provider_id,
            total_results=total_results,
            sample_size=sample_size,
            sample_indices=sample_indices,
            passed_count=passed_count,
            failed_count=failed_count,
            quality_scores=quality_scores,
            average_quality=average_quality,
            meets_threshold=meets_threshold,
            threshold=threshold,
            failure_reasons=failure_reasons,
            verification_method="sample",
        )

        logger.info(
            "Quality verification for RFQ %s (provider %s): %.2f avg quality, %d/%d passed",
            rfq_id, provider_id, average_quality, passed_count, sample_size
        )

        if not meets_threshold:
            logger.warning(
                "Quality below threshold for RFQ %s: %.2f < %.2f",
                rfq_id, average_quality, threshold
            )

        return report

    def verify_full(
        self,
        rfq_id: str,
        provider_id: str,
        results: list[Any],
        ground_truth: list[Any],
        quality_fn: Callable[[Any, Any], float],
        threshold: float = 0.85,
    ) -> QualityVerificationReport:
        """Verify ALL results (not sampled).

        Args:
            rfq_id: RFQ identifier
            provider_id: Provider who produced results
            results: Results to verify
            ground_truth: Ground truth for comparison
            quality_fn: Function to compute quality score
            threshold: Minimum acceptable quality

        Returns:
            Quality verification report
        """
        total_results = len(results)
        quality_scores = []
        passed_count = 0
        failed_count = 0
        failure_reasons = []

        for idx in range(total_results):
            result = results[idx]
            truth = ground_truth[idx] if idx < len(ground_truth) else None

            if truth is None:
                failed_count += 1
                failure_reasons.append(f"Result {idx}: no ground truth")
                continue

            try:
                score = quality_fn(result, truth)
                quality_scores.append(score)

                if score >= threshold:
                    passed_count += 1
                else:
                    failed_count += 1
                    failure_reasons.append(
                        f"Result {idx}: score {score:.3f} < threshold {threshold}"
                    )
            except Exception as e:
                logger.error("Quality check failed for result %d: %s", idx, e)
                failed_count += 1
                failure_reasons.append(f"Result {idx}: error - {str(e)}")

        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        meets_threshold = average_quality >= threshold

        report = QualityVerificationReport(
            rfq_id=rfq_id,
            provider_id=provider_id,
            total_results=total_results,
            sample_size=total_results,
            sample_indices=list(range(total_results)),
            passed_count=passed_count,
            failed_count=failed_count,
            quality_scores=quality_scores,
            average_quality=average_quality,
            meets_threshold=meets_threshold,
            threshold=threshold,
            failure_reasons=failure_reasons[:10],  # Limit to first 10 failures
            verification_method="full",
        )

        logger.info(
            "Full quality verification for RFQ %s: %.2f avg quality, %d/%d passed",
            rfq_id, average_quality, passed_count, total_results
        )

        return report
