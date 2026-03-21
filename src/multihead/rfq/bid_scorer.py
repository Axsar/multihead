"""Bid Scorer: Rank marketplace bids based on multiple factors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Bid, RFQ

logger = logging.getLogger(__name__)

# Scoring weights (higher = more important)
_W_PRICE = 40            # Lower price is better
_W_QUALITY = 30          # Higher quality guarantee
_W_REPUTATION = 20       # Higher reputation score
_W_SPEED = 10            # Faster completion

_W_PRIVACY_MATCH = 15    # Privacy level matches requirement
_W_ENCRYPTION = 10       # Supports encryption when needed


class BidScorer:
    """Scores and ranks bids for RFQ selection.

    Similar to Router scoring but for marketplace providers.
    """

    def score_bid(self, bid: Bid, rfq: RFQ) -> float:
        """Score a bid for an RFQ (higher is better).

        Args:
            bid: Bid to score
            rfq: RFQ being bid on

        Returns:
            Score (0.0-100.0+)
        """
        score = 0.0

        # Factor 1: Price (40 points) - lower is better
        if rfq.max_total_price and rfq.max_total_price > 0:
            # Normalize: price at max budget = 0 points, free = full points
            price_ratio = bid.total_price / rfq.max_total_price
            price_score = max(0.0, 1.0 - price_ratio)
            score += _W_PRICE * price_score
        elif rfq.max_price_per_unit and rfq.max_price_per_unit > 0:
            price_ratio = bid.price_per_unit / rfq.max_price_per_unit
            price_score = max(0.0, 1.0 - price_ratio)
            score += _W_PRICE * price_score
        else:
            # No budget specified, prefer cheaper
            # Assume $1/unit is expensive, $0 is free
            price_score = max(0.0, 1.0 - bid.price_per_unit)
            score += _W_PRICE * price_score

        # Factor 2: Quality (30 points) - higher guarantees = better
        quality_score = 0.0
        if bid.quality_guarantees:
            # Use highest quality guarantee
            quality_scores = [qg.min_value for qg in bid.quality_guarantees]
            quality_score = max(quality_scores)
        else:
            # No quality guarantee = assume moderate (0.7)
            quality_score = 0.7

        score += _W_QUALITY * quality_score

        # Factor 3: Reputation (20 points)
        reputation_score = bid.reputation_score  # Already 0.0-1.0
        score += _W_REPUTATION * reputation_score

        # Factor 4: Speed (10 points) - faster is better
        if rfq.deadline_hours and bid.estimated_time_hours > 0:
            # Normalize: at deadline = 0 points, instant = full points
            time_ratio = bid.estimated_time_hours / rfq.deadline_hours
            speed_score = max(0.0, 1.0 - time_ratio)
            score += _W_SPEED * speed_score
        elif bid.estimated_time_hours > 0:
            # No deadline, prefer faster (assume 24h is slow, 1h is fast)
            speed_score = max(0.0, 1.0 - bid.estimated_time_hours / 24.0)
            score += _W_SPEED * speed_score
        else:
            # No time estimate = moderate
            score += _W_SPEED * 0.5

        # Factor 5: Privacy match (15 points)
        privacy_score = 0.0
        if rfq.data_sensitivity == "confidential":
            # Confidential data requires local processing (but we wouldn't create RFQ for this)
            # If we're here, something went wrong
            privacy_score = 0.0
        elif rfq.data_sensitivity == "internal":
            # Internal data prefers encrypted
            if bid.privacy_level == "encrypted":
                privacy_score = 1.0
            elif bid.privacy_level == "local":
                privacy_score = 0.8  # Local is OK too
            else:
                privacy_score = 0.3  # External is risky
        else:
            # Public data - any privacy level OK
            privacy_score = 1.0

        score += _W_PRIVACY_MATCH * privacy_score

        # Factor 6: Encryption support (10 points) - when required
        if rfq.require_encryption:
            if bid.supports_encryption:
                score += _W_ENCRYPTION
            else:
                # Encryption required but not supported = disqualified
                return 0.0
        else:
            # Encryption not required, but nice to have
            if bid.supports_encryption:
                score += _W_ENCRYPTION * 0.5

        return score

    def rank_bids(self, bids: list[Bid], rfq: RFQ) -> list[tuple[Bid, float]]:
        """Rank all bids for an RFQ.

        Args:
            bids: List of bids to rank
            rfq: RFQ being bid on

        Returns:
            List of (bid, score) tuples sorted by score descending
        """
        scored = [(bid, self.score_bid(bid, rfq)) for bid in bids]
        scored.sort(key=lambda x: x[1], reverse=True)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Ranked %d bids for RFQ %s:", len(bids), rfq.rfq_id)
            for bid, score in scored:
                logger.debug(
                    "  %s: %.1f (price=$%.2f, quality=%.2f, reputation=%.2f)",
                    bid.provider_name, score, bid.total_price,
                    max([qg.min_value for qg in bid.quality_guarantees], default=0.0),
                    bid.reputation_score,
                )

        return scored

    def select_best_bid(self, bids: list[Bid], rfq: RFQ) -> Bid | None:
        """Select the best bid for an RFQ.

        Args:
            bids: List of bids to choose from
            rfq: RFQ being bid on

        Returns:
            Best bid or None if no valid bids
        """
        if not bids:
            return None

        ranked = self.rank_bids(bids, rfq)

        # Filter out bids that violate hard constraints
        valid_bids = []
        for bid, score in ranked:
            # Score 0 means disqualified
            if score == 0.0:
                continue

            # Check budget constraints
            if rfq.max_total_price and bid.total_price > rfq.max_total_price:
                logger.debug("Bid from %s exceeds budget", bid.provider_name)
                continue

            if rfq.max_price_per_unit and bid.price_per_unit > rfq.max_price_per_unit:
                logger.debug("Bid from %s exceeds per-unit budget", bid.provider_name)
                continue

            # Check deadline
            if rfq.deadline_hours and bid.estimated_time_hours > rfq.deadline_hours:
                logger.debug("Bid from %s exceeds deadline", bid.provider_name)
                continue

            # Check quality
            if rfq.min_quality_score > 0 and bid.quality_guarantees:
                max_quality = max([qg.min_value for qg in bid.quality_guarantees])
                if max_quality < rfq.min_quality_score:
                    logger.debug("Bid from %s below quality threshold", bid.provider_name)
                    continue

            valid_bids.append((bid, score))

        if not valid_bids:
            logger.warning("No valid bids for RFQ %s", rfq.rfq_id)
            return None

        # Return highest scoring valid bid
        best_bid, best_score = valid_bids[0]
        logger.info(
            "Selected bid from %s (score=%.1f, price=$%.2f, time=%.1fh)",
            best_bid.provider_name, best_score, best_bid.total_price,
            best_bid.estimated_time_hours,
        )

        return best_bid
