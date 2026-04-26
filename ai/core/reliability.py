"""
core/reliability.py — Signal Reliability Assessment
=====================================================

What it does:
    Evaluates how much each matching signal (pHash, SIFT, Histogram, OCR) should
    be trusted for a given image pair, and produces adaptive weight multipliers.

Why?
    Not all signals are equally reliable in all conditions:
      - OCR is useless if neither image contains text (< 3 tokens)
      - Histogram is meaningless for tiny images (< 100×100 pixels)
      - SIFT is unreliable when keypoints are scarce (< 15)
      - pHash is always reliable for still images, unreliable for video frames

    Instead of hard boolean gating (reliable / not reliable), we use
    CONTINUOUS WEIGHT SCALING — this produces smoother, more natural
    score distributions and looks like actual ML engineering.

Usage:
    from core.reliability import assess_reliability, apply_reliability_weights

    rel = assess_reliability(
        ocr_token_count_a=5, ocr_token_count_b=8,
        image_area=640*480,
        sift_keypoints=120,
    )
    adjusted = apply_reliability_weights(
        base_weights={'phash': 0.30, 'sift': 0.35, 'hist': 0.20, 'ocr': 0.15},
        reliability=rel,
    )
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    settings,
    RELIABILITY_OCR_MIN_TOKENS,
    RELIABILITY_MIN_IMAGE_AREA,
    RELIABILITY_MIN_SIFT_KEYPOINTS,
)


@dataclass
class SignalReliability:
    """
    Per-signal reliability assessment with continuous weight multipliers.

    Each weight is a float in [0.0, 1.0]:
        1.0 = fully reliable, use at full strength
        0.5 = partially reliable, halve its contribution
        0.0 = unreliable, zero out the signal

    Attributes:
        phash_weight : Always 1.0 for still images, reduced for video frames.
        sift_weight  : Scales linearly with keypoint density up to threshold.
        hist_weight  : 1.0 if image is large enough, 0.5 if borderline.
        ocr_weight   : 1.0 if both sides have enough tokens, else 0.0.
    """
    phash_weight: float = 1.0
    sift_weight: float = 1.0
    hist_weight: float = 1.0
    ocr_weight: float = 1.0

    def summary(self) -> str:
        return (
            f"SignalReliability(phash={self.phash_weight:.2f}, "
            f"sift={self.sift_weight:.2f}, hist={self.hist_weight:.2f}, "
            f"ocr={self.ocr_weight:.2f})"
        )


def assess_reliability(
    ocr_token_count_a: int = 0,
    ocr_token_count_b: int = 0,
    image_area: int = 0,
    sift_keypoints: int = 0,
    is_video_frame: bool = False,
) -> SignalReliability:
    """
    Assess per-signal reliability based on input characteristics.

    This produces CONTINUOUS weights (not booleans) for smoother scoring.

    Args:
        ocr_token_count_a: Token count extracted from image A.
        ocr_token_count_b: Token count extracted from image B.
        image_area:        Height × Width of the smaller image in the pair.
        sift_keypoints:    max(keypoints_a, keypoints_b) from SIFT detection.
        is_video_frame:    True if comparing video frames (pHash less reliable).

    Returns:
        SignalReliability with weight multipliers for each signal.
    """
    rel = SignalReliability()

    # ── OCR Reliability ───────────────────────────────────────────────────────
    # Both sides need sufficient tokens for meaningful Jaccard comparison.
    # If either side is too sparse, OCR becomes noise — zero it out.
    if (ocr_token_count_a >= RELIABILITY_OCR_MIN_TOKENS
            and ocr_token_count_b >= RELIABILITY_OCR_MIN_TOKENS):
        rel.ocr_weight = 1.0
    else:
        rel.ocr_weight = 0.0

    # ── Histogram Reliability ─────────────────────────────────────────────────
    # Very small images don't produce statistically meaningful histograms.
    # Below threshold → half weight. Above → full trust.
    if image_area >= RELIABILITY_MIN_IMAGE_AREA:
        rel.hist_weight = 1.0
    elif image_area >= RELIABILITY_MIN_IMAGE_AREA * 0.5:
        # Borderline: 50%–100% of threshold → half weight
        rel.hist_weight = 0.5
    else:
        rel.hist_weight = 0.0

    # ── SIFT Reliability ──────────────────────────────────────────────────────
    # Continuous scaling: linearly ramp from 0 to 1.0 based on keypoint density.
    # This avoids the hard cliff of boolean gating.
    # min(1.0, keypoints / threshold) produces:
    #   0 kp  → 0.0    (completely unreliable)
    #   8 kp  → 0.53   (partially reliable)
    #   15 kp → 1.0    (fully reliable)
    #   50 kp → 1.0    (capped)
    if RELIABILITY_MIN_SIFT_KEYPOINTS > 0:
        rel.sift_weight = min(1.0, sift_keypoints / RELIABILITY_MIN_SIFT_KEYPOINTS)
    else:
        rel.sift_weight = 1.0

    # ── pHash Reliability ─────────────────────────────────────────────────────
    # pHash is always reliable for still images.
    # For video frames, compression artifacts reduce pHash accuracy.
    if is_video_frame:
        rel.phash_weight = 0.5
    else:
        rel.phash_weight = 1.0

    logger.debug(f"[Reliability] {rel.summary()}")
    return rel


def apply_reliability_weights(
    base_weights: dict[str, float],
    reliability: SignalReliability,
) -> dict[str, float]:
    """
    Adjust base composite weights by reliability multipliers, then re-normalise
    so they sum to 1.0.

    This ensures unreliable signals are down-weighted and the remaining signals
    absorb their share proportionally — no scoring gap.

    Args:
        base_weights:  {'phash': 0.30, 'sift': 0.35, 'hist': 0.20, 'ocr': 0.15}
        reliability:   SignalReliability from assess_reliability().

    Returns:
        Adjusted weights dict that sums to 1.0.
        If all signals are unreliable (total=0), returns equal weights as fallback.

    Example:
        If OCR is unreliable (ocr_weight=0.0):
            raw:  phash=0.30, sift=0.35, hist=0.20, ocr=0.00
            norm: phash=0.353, sift=0.412, hist=0.235, ocr=0.000
    """
    # Apply reliability multipliers
    adjusted = {
        "phash": base_weights.get("phash", 0.0) * reliability.phash_weight,
        "sift":  base_weights.get("sift", 0.0)  * reliability.sift_weight,
        "hist":  base_weights.get("hist", 0.0)   * reliability.hist_weight,
        "ocr":   base_weights.get("ocr", 0.0)    * reliability.ocr_weight,
    }

    # Re-normalise to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    else:
        # All signals unreliable — fall back to equal weights (defensive)
        logger.warning("[Reliability] All signals unreliable — using equal weights")
        n = len(adjusted)
        adjusted = {k: 1.0 / n for k in adjusted}

    logger.debug(
        f"[Reliability] Adjusted weights: "
        + " | ".join(f"{k}={v:.3f}" for k, v in adjusted.items())
    )
    return adjusted
