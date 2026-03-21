"""RFQ Manager: Create RFQs, collect bids, select providers."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from multihead.models import BudgetConstraint, PrivacyConstraint, StepDef

from .models import Bid, RFQ, RFQStatus

logger = logging.getLogger(__name__)


class RFQManager:
    """Manages RFQ lifecycle: create, publish, collect bids, award."""

    def __init__(self, project_id: str | None = None):
        """Initialize RFQ manager.

        Args:
            project_id: Optional project ID for tracking
        """
        self.project_id = project_id
        self.rfqs: dict[str, RFQ] = {}  # rfq_id -> RFQ

    def create_rfq_from_step(
        self,
        step: StepDef,
        data_count: int = 1,
        sample_data: list[Any] | None = None,
    ) -> RFQ:
        """Create an RFQ from a step definition.

        Args:
            step: Step definition with task requirements
            data_count: Number of items to process
            sample_data: Optional sample data for providers

        Returns:
            RFQ ready to publish
        """
        rfq_id = f"rfq-{uuid.uuid4().hex[:12]}"

        # Extract task type (use first task_type)
        task_type = step.task_types[0] if step.task_types else "unknown"

        # Build task description
        description = f"Execute {task_type} on {data_count} items"
        if step.name:
            description = f"{step.name}: {description}"

        # Extract budget constraints
        budget = step.budget
        max_price_per_unit = budget.max_cost_per_step if budget else None
        max_total_price = budget.max_total_cost if budget else None

        # Extract privacy constraints
        privacy = step.privacy
        data_sensitivity = privacy.data_sensitivity.value if privacy else "public"
        require_encryption = privacy.require_encryption if privacy else False
        allowed_providers = privacy.allowed_providers or [] if privacy else []
        blocked_providers = privacy.blocked_providers or [] if privacy else []

        # Extract timing
        deadline_hours = None
        if budget and budget.max_total_time_s:
            deadline_hours = budget.max_total_time_s / 3600.0

        # Create RFQ
        rfq = RFQ(
            rfq_id=rfq_id,
            task_type=task_type,
            task_description=description,
            data_count=data_count,
            sample_data=sample_data or [],
            required_capabilities=step.task_types,
            max_price_per_unit=max_price_per_unit,
            max_total_price=max_total_price,
            data_sensitivity=data_sensitivity,
            require_encryption=require_encryption,
            allowed_providers=allowed_providers,
            blocked_providers=blocked_providers,
            deadline_hours=deadline_hours,
            project_id=self.project_id,
        )

        self.rfqs[rfq_id] = rfq
        logger.info("Created RFQ %s for %s (%d items)", rfq_id, task_type, data_count)

        return rfq

    def publish_rfq(
        self,
        rfq_id: str,
        bidding_window_hours: float = 1.0,
    ) -> RFQ:
        """Publish an RFQ to the marketplace.

        Args:
            rfq_id: RFQ to publish
            bidding_window_hours: How long to accept bids

        Returns:
            Published RFQ
        """
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        rfq.publish(bidding_window_hours)
        logger.info(
            "Published RFQ %s (bidding ends: %s)",
            rfq_id, rfq.bidding_ends_at
        )

        return rfq

    def add_bid(self, rfq_id: str, bid: Bid) -> None:
        """Add a bid to an RFQ.

        Args:
            rfq_id: RFQ receiving the bid
            bid: Bid to add
        """
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        # Validate against privacy constraints
        if rfq.blocked_providers and bid.provider_id in rfq.blocked_providers:
            logger.warning("Bid from blocked provider %s rejected", bid.provider_id)
            return

        if rfq.allowed_providers and bid.provider_id not in rfq.allowed_providers:
            logger.warning("Bid from non-whitelisted provider %s rejected", bid.provider_id)
            return

        # Validate encryption requirement
        if rfq.require_encryption and not bid.supports_encryption:
            logger.warning("Bid from %s rejected (encryption required)", bid.provider_id)
            return

        rfq.add_bid(bid)
        logger.info(
            "Added bid from %s to RFQ %s (price: $%.4f/unit, total: $%.2f)",
            bid.provider_name, rfq_id, bid.price_per_unit, bid.total_price
        )

    def get_bids(self, rfq_id: str) -> list[Bid]:
        """Get all bids for an RFQ.

        Args:
            rfq_id: RFQ to query

        Returns:
            List of bids
        """
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            return []

        return rfq.bids

    def select_bid(self, rfq_id: str, bid_id: str) -> Bid:
        """Award RFQ to a specific bid.

        Args:
            rfq_id: RFQ to award
            bid_id: Winning bid

        Returns:
            Selected bid
        """
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        rfq.select_bid(bid_id)
        bid = rfq.get_selected_bid()

        logger.info(
            "Awarded RFQ %s to %s (total: $%.2f, time: %.1fh)",
            rfq_id, bid.provider_name, bid.total_price, bid.estimated_time_hours
        )

        return bid

    def get_rfq(self, rfq_id: str) -> RFQ | None:
        """Get an RFQ by ID.

        Args:
            rfq_id: RFQ identifier

        Returns:
            RFQ or None if not found
        """
        return self.rfqs.get(rfq_id)

    def list_rfqs(
        self,
        status: RFQStatus | None = None,
    ) -> list[RFQ]:
        """List all RFQs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of RFQs
        """
        rfqs = list(self.rfqs.values())

        if status:
            rfqs = [r for r in rfqs if r.status == status]

        # Sort by creation time (newest first)
        rfqs.sort(key=lambda r: r.created_at, reverse=True)

        return rfqs

    def close_bidding(self, rfq_id: str) -> int:
        """Close bidding window for an RFQ.

        Args:
            rfq_id: RFQ to close

        Returns:
            Number of bids received
        """
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        if rfq.status != RFQStatus.BIDDING:
            logger.warning("RFQ %s not in bidding state", rfq_id)
            return len(rfq.bids)

        # Just mark as published (no longer accepting bids)
        rfq.status = RFQStatus.PUBLISHED

        logger.info("Closed bidding for RFQ %s (%d bids received)", rfq_id, len(rfq.bids))

        return len(rfq.bids)

    def cancel_rfq(self, rfq_id: str, reason: str = "") -> None:
        """Cancel an RFQ.

        Args:
            rfq_id: RFQ to cancel
            reason: Cancellation reason
        """
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        rfq.status = RFQStatus.CANCELLED
        rfq.notes = f"Cancelled: {reason}" if reason else "Cancelled"

        logger.info("Cancelled RFQ %s: %s", rfq_id, reason or "no reason given")
