"""Data models for RFQ (Request for Quote) workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RFQStatus(str, Enum):
    """RFQ lifecycle status."""
    DRAFT = "draft"
    PUBLISHED = "published"
    BIDDING = "bidding"
    AWARDED = "awarded"
    CANCELLED = "cancelled"


class QualityGuarantee(BaseModel):
    """Quality guarantee offered by a provider."""
    metric: str  # "accuracy", "mAP", "F1", etc.
    min_value: float  # Minimum guaranteed value (0.0-1.0)
    verification_method: str = "sample"  # "sample", "full", "none"


class Bid(BaseModel):
    """Bid from a BotVibes provider."""
    bid_id: str
    rfq_id: str
    provider_id: str
    provider_name: str

    # Pricing
    price_per_unit: float  # USD per task unit
    total_price: float  # Total cost for full task

    # Performance
    estimated_time_hours: float
    quality_guarantees: list[QualityGuarantee] = Field(default_factory=list)

    # Provider reputation
    reputation_score: float = 0.0  # 0.0-1.0
    completed_tasks: int = 0
    success_rate: float = 0.0  # 0.0-1.0

    # Privacy
    supports_encryption: bool = False
    privacy_level: str = "external"  # "local", "encrypted", "external"

    # Metadata
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    notes: str = ""


class RFQ(BaseModel):
    """Request for Quote to BotVibes marketplace."""
    rfq_id: str

    # Task specification
    task_type: str  # "object_detection", "segmentation", etc.
    task_description: str
    data_count: int  # Number of items to process
    sample_data: list[Any] = Field(default_factory=list)  # Optional samples

    # Requirements
    required_capabilities: list[str] = Field(default_factory=list)
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)

    # Budget constraints
    max_price_per_unit: float | None = None  # USD per unit
    max_total_price: float | None = None  # Total budget cap

    # Quality requirements
    min_quality_score: float = 0.0  # Minimum acceptable quality
    quality_metric: str = "accuracy"  # Metric to use

    # Privacy constraints
    data_sensitivity: str = "public"  # "public", "internal", "confidential"
    require_encryption: bool = False
    allowed_providers: list[str] = Field(default_factory=list)
    blocked_providers: list[str] = Field(default_factory=list)

    # Timing
    deadline_hours: float | None = None  # Max time to completion
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bidding_ends_at: datetime | None = None

    # Status
    status: RFQStatus = RFQStatus.DRAFT
    bids: list[Bid] = Field(default_factory=list)
    selected_bid_id: str | None = None

    # Metadata
    requester_id: str = "multihead"
    project_id: str | None = None
    notes: str = ""

    def publish(self, bidding_window_hours: float = 1.0) -> None:
        """Publish RFQ to marketplace and start bidding window."""
        self.status = RFQStatus.BIDDING
        self.bidding_ends_at = datetime.now(timezone.utc) + timedelta(hours=bidding_window_hours)

    def add_bid(self, bid: Bid) -> None:
        """Add a bid to the RFQ."""
        if self.status != RFQStatus.BIDDING:
            raise ValueError(f"Cannot add bid to RFQ with status {self.status}")

        if self.bidding_ends_at and datetime.now(timezone.utc) > self.bidding_ends_at:
            raise ValueError("Bidding window has closed")

        self.bids.append(bid)

    def select_bid(self, bid_id: str) -> None:
        """Award the RFQ to a specific bid."""
        if bid_id not in [b.bid_id for b in self.bids]:
            raise ValueError(f"Bid {bid_id} not found in RFQ")

        self.selected_bid_id = bid_id
        self.status = RFQStatus.AWARDED

    def get_selected_bid(self) -> Bid | None:
        """Get the selected bid, if any."""
        if not self.selected_bid_id:
            return None

        for bid in self.bids:
            if bid.bid_id == self.selected_bid_id:
                return bid

        return None


class DelegationResult(BaseModel):
    """Result from delegating a task to marketplace."""
    rfq_id: str
    provider_id: str
    task_id: str  # ACP task ID

    # Execution
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"  # "running", "completed", "failed"

    # Results
    output_data: Any = None
    error_message: str | None = None

    # Quality verification
    quality_verified: bool = False
    quality_score: float | None = None
    sample_verified_count: int = 0
    sample_failed_count: int = 0

    # Cost
    actual_cost: float | None = None
    actual_time_hours: float | None = None

    # Feedback
    accepted: bool = False
    feedback_submitted: bool = False
    reputation_delta: float = 0.0  # Change in provider reputation


class QualityVerificationReport(BaseModel):
    """Report from quality verification process."""
    rfq_id: str
    provider_id: str

    # Sampling
    total_results: int
    sample_size: int
    sample_indices: list[int] = Field(default_factory=list)

    # Verification results
    passed_count: int = 0
    failed_count: int = 0
    quality_scores: list[float] = Field(default_factory=list)

    # Overall
    average_quality: float = 0.0
    meets_threshold: bool = False
    threshold: float = 0.0

    # Failures
    failure_reasons: list[str] = Field(default_factory=list)

    # Metadata
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verification_method: str = "sample"  # "sample", "full"
