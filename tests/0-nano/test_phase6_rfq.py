"""Tests for Phase 6: RFQ (Request for Quote) system."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from multihead.models import BudgetConstraint, DataSensitivity, PrivacyConstraint, StepDef
from multihead.rfq import (
    Bid,
    BidScorer,
    QualityGuarantee,
    RFQ,
    RFQManager,
    RFQStatus,
)
from multihead.rfq.quality_verifier import QualityVerifier


@pytest.fixture
def rfq_manager():
    """Create RFQ manager."""
    return RFQManager(project_id="test-project")


@pytest.fixture
def sample_step():
    """Create sample step definition."""
    return StepDef(
        step_id="test-step",
        name="Detect comic panels",
        task_types=["object_detection", "comic_panel_detection"],
        budget=BudgetConstraint(
            max_cost_per_step=0.05,
            max_total_cost=500.00,
            max_total_time_s=12 * 3600,  # 12 hours
        ),
        privacy=PrivacyConstraint(
            data_sensitivity=DataSensitivity.INTERNAL,
            require_encryption=True,
        ),
    )


@pytest.fixture
def sample_bids():
    """Create sample bids from providers."""
    return [
        Bid(
            bid_id="bid-1",
            rfq_id="rfq-test",
            provider_id="provider-a",
            provider_name="Fast Vision Expert",
            price_per_unit=0.04,
            total_price=400.00,
            estimated_time_hours=4.0,
            quality_guarantees=[
                QualityGuarantee(metric="mAP", min_value=0.94, verification_method="sample")
            ],
            reputation_score=0.98,
            completed_tasks=150,
            success_rate=0.96,
            supports_encryption=True,
            privacy_level="encrypted",
        ),
        Bid(
            bid_id="bid-2",
            rfq_id="rfq-test",
            provider_id="provider-b",
            provider_name="Budget Detector",
            price_per_unit=0.02,
            total_price=200.00,
            estimated_time_hours=24.0,
            quality_guarantees=[
                QualityGuarantee(metric="mAP", min_value=0.88, verification_method="sample")
            ],
            reputation_score=0.88,
            completed_tasks=80,
            success_rate=0.92,
            supports_encryption=True,
            privacy_level="encrypted",
        ),
        Bid(
            bid_id="bid-3",
            rfq_id="rfq-test",
            provider_id="provider-c",
            provider_name="Premium Quality",
            price_per_unit=0.03,
            total_price=300.00,
            estimated_time_hours=8.0,
            quality_guarantees=[
                QualityGuarantee(metric="mAP", min_value=0.92, verification_method="sample")
            ],
            reputation_score=0.95,
            completed_tasks=200,
            success_rate=0.97,
            supports_encryption=True,
            privacy_level="encrypted",
        ),
    ]


class TestRFQModels:
    """Test RFQ data models."""

    def test_rfq_creation(self):
        """Test creating an RFQ."""
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Detect panels in 10k images",
            data_count=10000,
            max_total_price=500.00,
            deadline_hours=12.0,
        )

        assert rfq.rfq_id == "rfq-test"
        assert rfq.task_type == "object_detection"
        assert rfq.data_count == 10000
        assert rfq.status == RFQStatus.DRAFT
        assert len(rfq.bids) == 0

    def test_rfq_publish(self):
        """Test publishing an RFQ."""
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Test",
            data_count=100,
        )

        rfq.publish(bidding_window_hours=2.0)

        assert rfq.status == RFQStatus.BIDDING
        assert rfq.bidding_ends_at is not None
        # Should be ~2 hours from now
        delta = (rfq.bidding_ends_at - datetime.now(timezone.utc)).total_seconds()
        assert 7000 < delta < 7300  # ~2 hours (allowing for test execution time)

    def test_rfq_add_bid(self):
        """Test adding bids to an RFQ."""
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Test",
            data_count=100,
        )
        rfq.publish()

        bid = Bid(
            bid_id="bid-1",
            rfq_id="rfq-test",
            provider_id="provider-a",
            provider_name="Test Provider",
            price_per_unit=0.05,
            total_price=500.00,
            estimated_time_hours=10.0,
        )

        rfq.add_bid(bid)

        assert len(rfq.bids) == 1
        assert rfq.bids[0].bid_id == "bid-1"

    def test_rfq_select_bid(self):
        """Test selecting a winning bid."""
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Test",
            data_count=100,
        )
        rfq.publish()

        bid = Bid(
            bid_id="bid-1",
            rfq_id="rfq-test",
            provider_id="provider-a",
            provider_name="Test Provider",
            price_per_unit=0.05,
            total_price=500.00,
            estimated_time_hours=10.0,
        )

        rfq.add_bid(bid)
        rfq.select_bid("bid-1")

        assert rfq.status == RFQStatus.AWARDED
        assert rfq.selected_bid_id == "bid-1"
        assert rfq.get_selected_bid().bid_id == "bid-1"


class TestRFQManager:
    """Test RFQ Manager."""

    def test_create_rfq_from_step(self, rfq_manager, sample_step):
        """Test creating RFQ from step definition."""
        rfq = rfq_manager.create_rfq_from_step(
            step=sample_step,
            data_count=10000,
        )

        assert rfq.task_type == "object_detection"
        assert rfq.data_count == 10000
        assert rfq.max_price_per_unit == 0.05  # Extracted from budget.max_cost_per_step
        assert rfq.max_total_price == 500.00
        assert rfq.data_sensitivity == "internal"
        assert rfq.require_encryption is True
        assert rfq.deadline_hours == 12.0
        assert rfq.project_id == "test-project"

    def test_publish_rfq(self, rfq_manager, sample_step):
        """Test publishing an RFQ."""
        rfq = rfq_manager.create_rfq_from_step(sample_step, data_count=100)
        rfq_manager.publish_rfq(rfq.rfq_id, bidding_window_hours=1.0)

        rfq = rfq_manager.get_rfq(rfq.rfq_id)
        assert rfq.status == RFQStatus.BIDDING

    def test_add_bid(self, rfq_manager, sample_step, sample_bids):
        """Test adding bids to an RFQ."""
        rfq = rfq_manager.create_rfq_from_step(sample_step, data_count=100)
        rfq_manager.publish_rfq(rfq.rfq_id)

        # Add first bid
        rfq_manager.add_bid(rfq.rfq_id, sample_bids[0])

        bids = rfq_manager.get_bids(rfq.rfq_id)
        assert len(bids) == 1

    def test_add_bid_respects_privacy_constraints(self, rfq_manager, sample_step):
        """Test that bids without encryption are rejected."""
        rfq = rfq_manager.create_rfq_from_step(sample_step, data_count=100)
        rfq_manager.publish_rfq(rfq.rfq_id)

        # Create bid without encryption support
        bid = Bid(
            bid_id="bid-no-encrypt",
            rfq_id=rfq.rfq_id,
            provider_id="provider-x",
            provider_name="No Encryption Provider",
            price_per_unit=0.01,
            total_price=100.00,
            estimated_time_hours=5.0,
            supports_encryption=False,  # RFQ requires encryption!
        )

        rfq_manager.add_bid(rfq.rfq_id, bid)

        # Bid should be rejected
        bids = rfq_manager.get_bids(rfq.rfq_id)
        assert len(bids) == 0

    def test_select_bid(self, rfq_manager, sample_step, sample_bids):
        """Test selecting a winning bid."""
        rfq = rfq_manager.create_rfq_from_step(sample_step, data_count=100)
        rfq_manager.publish_rfq(rfq.rfq_id)

        for bid in sample_bids:
            rfq_manager.add_bid(rfq.rfq_id, bid)

        # Select first bid
        selected = rfq_manager.select_bid(rfq.rfq_id, sample_bids[0].bid_id)

        assert selected.bid_id == sample_bids[0].bid_id
        rfq = rfq_manager.get_rfq(rfq.rfq_id)
        assert rfq.status == RFQStatus.AWARDED


class TestBidScorer:
    """Test Bid Scorer."""

    def test_score_bid(self, sample_bids):
        """Test scoring a single bid."""
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Test",
            data_count=10000,
            max_total_price=500.00,
            deadline_hours=12.0,
            data_sensitivity="internal",
            require_encryption=True,
        )

        scorer = BidScorer()
        score = scorer.score_bid(sample_bids[0], rfq)

        # Should get high score (good quality, good reputation, within budget)
        assert score > 50.0

    def test_rank_bids(self, sample_bids):
        """Test ranking multiple bids."""
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Test",
            data_count=10000,
            max_total_price=500.00,
            deadline_hours=12.0,
        )

        scorer = BidScorer()
        ranked = scorer.rank_bids(sample_bids, rfq)

        assert len(ranked) == 3
        # Should be sorted by score descending
        assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]

    def test_select_best_bid(self, sample_bids):
        """Test selecting best bid."""
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Test",
            data_count=10000,
            max_total_price=500.00,
            deadline_hours=12.0,
        )

        scorer = BidScorer()
        best = scorer.select_best_bid(sample_bids, rfq)

        assert best is not None
        # Scoring favors provider-c (good quality 0.92, best cost-to-quality ratio)
        # provider-a is faster but more expensive
        # provider-c balances quality (0.92) + reputation (0.95) + price ($300)
        assert best.provider_id == "provider-c"

    def test_select_best_bid_respects_budget(self, sample_bids):
        """Test that bids exceeding budget are filtered out."""
        # Set very low budget
        rfq = RFQ(
            rfq_id="rfq-test",
            task_type="object_detection",
            task_description="Test",
            data_count=10000,
            max_total_price=150.00,  # Only bid-2 ($200) is close but over
        )

        scorer = BidScorer()
        best = scorer.select_best_bid(sample_bids, rfq)

        # No bids should pass (all exceed budget)
        assert best is None


class TestQualityVerifier:
    """Test Quality Verifier."""

    def test_verify_results_with_ground_truth(self):
        """Test quality verification with ground truth."""
        verifier = QualityVerifier(sample_rate=0.2, min_samples=5)

        # Mock results and ground truth
        results = [f"result-{i}" for i in range(50)]
        ground_truth = [f"result-{i}" for i in range(50)]  # Perfect match

        def quality_fn(result, truth):
            return 1.0 if result == truth else 0.0

        report = verifier.verify_results(
            rfq_id="rfq-test",
            provider_id="provider-a",
            results=results,
            ground_truth=ground_truth,
            quality_fn=quality_fn,
            threshold=0.85,
        )

        assert report.total_results == 50
        assert report.sample_size == 10  # 20% of 50
        assert report.passed_count == 10  # All should pass
        assert report.average_quality == 1.0
        assert report.meets_threshold is True

    def test_verify_results_below_threshold(self):
        """Test quality verification when quality is below threshold."""
        verifier = QualityVerifier(sample_rate=0.2, min_samples=5)

        # Mock poor quality results
        results = [f"bad-result-{i}" for i in range(50)]
        ground_truth = [f"result-{i}" for i in range(50)]

        def quality_fn(result, truth):
            # Always return low quality
            return 0.5

        report = verifier.verify_results(
            rfq_id="rfq-test",
            provider_id="provider-bad",
            results=results,
            ground_truth=ground_truth,
            quality_fn=quality_fn,
            threshold=0.85,
        )

        assert report.average_quality == 0.5
        assert report.meets_threshold is False
        assert len(report.failure_reasons) > 0

    def test_verify_full(self):
        """Test full verification (no sampling)."""
        verifier = QualityVerifier()

        results = [f"result-{i}" for i in range(20)]
        ground_truth = [f"result-{i}" for i in range(20)]

        def quality_fn(result, truth):
            return 1.0 if result == truth else 0.0

        report = verifier.verify_full(
            rfq_id="rfq-test",
            provider_id="provider-a",
            results=results,
            ground_truth=ground_truth,
            quality_fn=quality_fn,
            threshold=0.85,
        )

        assert report.total_results == 20
        assert report.sample_size == 20  # Full verification
        assert report.verification_method == "full"
        assert report.average_quality == 1.0
