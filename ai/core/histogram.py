"""
core/histogram.py — HSV Histogram Comparison (Color Similarity)
================================================================

What it does:
    Compares the colour distribution of two images using HSV histograms
    and Chi-square distance.

Why HSV?
    Unlike RGB, HSV separates colour (Hue) from brightness (Value).
    This makes comparison robust to lighting changes — a photo taken in
    daylight and shadow will have similar H/S distributions even if V
    differs significantly.

Algorithm:
    1. Convert image BGR → HSV
    2. Compute a 3D histogram over [H, S, V] channels with configurable bins
    3. Normalise histogram so total counts sum to 1.0
    4. Compare two histograms using Chi-square distance
    5. Convert Chi-square distance → similarity via exponential decay:
           similarity = exp(−chi2 / HISTOGRAM_NORMALIZATION_FACTOR)
       This maps [0, +∞) → (0, 1]:
           chi2 = 0     → similarity = 1.0 (identical)
           chi2 → ∞     → similarity → 0.0 (completely different)

Usage:
    from core.histogram import HistogramMatcher
    hm = HistogramMatcher()
    score = hm.compare("img_a.jpg", "img_b.jpg")
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    settings,
    HISTOGRAM_NORMALIZATION_FACTOR,
    HISTOGRAM_MAX_SIMILARITY,
    HISTOGRAM_SIFT_DAMPENING_THRESHOLD,
)


class HistogramMatcher:
    """
    Computes and compares HSV colour histograms for image similarity.

    Args:
        bins       (int):   Number of bins per HSV channel.
        threshold  (float): Max chi-square distance to consider 'similar'.
        norm_factor(float): Denominator in exp(−chi2 / norm_factor).
    """

    def __init__(
        self,
        bins: int = settings.HIST_BINS,
        threshold: float = settings.HIST_THRESHOLD,
        norm_factor: float = HISTOGRAM_NORMALIZATION_FACTOR,
    ) -> None:
        self.bins = bins
        self.threshold = threshold
        self.norm_factor = norm_factor
        logger.debug(
            f"[Histogram] Initialised | bins={bins} threshold={threshold} "
            f"norm_factor={norm_factor}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compare(
        self,
        source_a: Union[str, Path, np.ndarray],
        source_b: Union[str, Path, np.ndarray],
        sift_score: Optional[float] = None,
    ) -> float:
        """
        Compare two images by HSV histogram similarity.

        Args:
            source_a:    File path or BGR NumPy array for image A.
            source_b:    File path or BGR NumPy array for image B.
            sift_score:  Optional SIFT score for cross-signal dampening.
                         When provided and < HISTOGRAM_SIFT_DAMPENING_THRESHOLD,
                         histogram similarity is halved (same colours ≠ same content).

        Returns:
            Similarity score in [0.0, 1.0], clamped at HISTOGRAM_MAX_SIMILARITY.
            Returns 0.0 on any error (logged).
        """
        hist_a = self.compute_histogram(source_a)
        hist_b = self.compute_histogram(source_b)

        if hist_a is None or hist_b is None:
            logger.warning("[Histogram] Cannot compare — one or both histograms failed")
            return 0.0

        chi2 = self.chi_square_distance(hist_a, hist_b)
        similarity = self.chi2_to_similarity(chi2)

        # ── Clamp: histogram is a SUPPORTING signal, not dominant ─────────────
        # Same colours don't guarantee same content — cap prevents inflation
        similarity = min(similarity, HISTOGRAM_MAX_SIMILARITY)

        # ── SIFT-conditional dampening ────────────────────────────────────────
        # If SIFT found very few structural matches, distrust colour-only similarity
        # This catches the "same palette, different object" false-positive pattern
        if sift_score is not None and sift_score < HISTOGRAM_SIFT_DAMPENING_THRESHOLD:
            similarity *= 0.5
            logger.debug(
                f"[Histogram] SIFT dampening applied — sift_score={sift_score:.4f} "
                f"< {HISTOGRAM_SIFT_DAMPENING_THRESHOLD} → similarity halved"
            )

        logger.debug(
            f"[Histogram] Comparison complete | chi2={chi2:.4f} "
            f"similarity={similarity:.4f} (capped at {HISTOGRAM_MAX_SIMILARITY})"
        )
        return similarity

    def compute_histogram(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Compute a normalised 3D HSV histogram for an image.

        Steps:
            1. Load image as BGR (OpenCV native)
            2. Convert BGR → HSV
            3. Compute 3D histogram [H bins × S bins × V bins]
            4. Normalise so all bin values sum to 1.0

        Args:
            source: File path or BGR NumPy array.

        Returns:
            Flattened, normalised float32 histogram, or None on error.
        """
        img = self._load_bgr(source)
        if img is None:
            return None

        try:
            # Convert BGR → HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # ── Build the 3D histogram ────────────────────────────────────────
            # channels=[0, 1, 2] → H, S, V
            # histSize=[bins, bins, bins] → bins per channel
            # ranges:
            #   Hue:        [0, 180)  — OpenCV uses 0–179 for H
            #   Saturation: [0, 256)  — 0–255
            #   Value:      [0, 256)  — 0–255
            hist = cv2.calcHist(
                images=[hsv],
                channels=[0, 1, 2],
                mask=None,                              # no masking
                histSize=[self.bins, self.bins, self.bins],
                ranges=[0, 180, 0, 256, 0, 256],
            )

            # Normalise so total probability sums to 1.0
            # This makes histograms comparable across different image sizes.
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

            # Flatten to 1D for cv2.compareHist()
            hist_flat = hist.flatten().astype(np.float32)

            logger.debug(
                f"[Histogram] Computed histogram | shape={hist.shape} "
                f"bins_total={hist_flat.shape[0]}"
            )
            return hist_flat

        except Exception as exc:
            logger.error(f"[Histogram] histogram computation failed: {exc}")
            return None

    def chi_square_distance(
        self,
        hist_a: np.ndarray,
        hist_b: np.ndarray,
    ) -> float:
        """
        Compute Chi-square distance between two normalised histograms.

        Formula (per bin):
            χ²(a, b) = Σ  (a_i − b_i)² / (a_i + b_i)
                       i  where (a_i + b_i) > 0

        Properties:
            - χ² = 0   → identical distributions
            - χ² → ∞   → completely different distributions
            - Symmetric: χ²(a,b) == χ²(b,a)

        OpenCV cv2.HISTCMP_CHISQR implements this formula natively.

        Args:
            hist_a: Normalised histogram of image A.
            hist_b: Normalised histogram of image B.

        Returns:
            Chi-square distance ≥ 0.
        """
        # Ensure the histograms are the same size (defensive check)
        if hist_a.shape != hist_b.shape:
            logger.error(
                f"[Histogram] Histogram shape mismatch: "
                f"{hist_a.shape} vs {hist_b.shape}"
            )
            return float("inf")

        # cv2.compareHist expects (N,1) or (N,) float32 arrays
        distance = cv2.compareHist(
            hist_a.astype(np.float32),
            hist_b.astype(np.float32),
            cv2.HISTCMP_CHISQR,
        )

        logger.debug(f"[Histogram] Chi-square distance = {distance:.6f}")
        return float(distance)

    def chi2_to_similarity(self, chi2: float) -> float:
        """
        Convert Chi-square distance to a [0, 1] similarity score via
        exponential decay.

        Formula:
            similarity = exp(−chi2 / HISTOGRAM_NORMALIZATION_FACTOR)

        Behaviour:
            chi2 = 0       → exp(0)        = 1.00  (identical)
            chi2 = 100     → exp(−1)       ≈ 0.37  (moderate)
            chi2 = 500     → exp(−5)       ≈ 0.01  (very different)
            chi2 = +inf    → exp(−∞)       = 0.00

        The norm_factor (default=100) controls sensitivity:
            Higher → more forgiving (larger distances still score high)
            Lower  → stricter (distances above norm_factor drop quickly)

        Args:
            chi2: Non-negative chi-square distance.

        Returns:
            Float in [0.0, 1.0].
        """
        if chi2 < 0:
            # Shouldn't happen, but defensive
            chi2 = 0.0

        if self.norm_factor <= 0:
            logger.warning("[Histogram] norm_factor must be > 0; defaulting to 100")
            self.norm_factor = 100.0

        similarity = math.exp(-chi2 / self.norm_factor)

        # Clamp to [0, 1] (exp already returns (0, 1] but be safe)
        similarity = max(0.0, min(1.0, similarity))

        logger.debug(
            f"[Histogram] chi2_to_similarity: chi2={chi2:.4f} "
            f"norm_factor={self.norm_factor} → similarity={similarity:.4f}"
        )
        return similarity

    def is_color_similar(
        self,
        source_a: Union[str, Path, np.ndarray],
        source_b: Union[str, Path, np.ndarray],
    ) -> Tuple[bool, float]:
        """
        Convenience method: compare + threshold check.

        Returns:
            (is_similar, similarity_score)
        """
        score = self.compare(source_a, source_b)
        is_similar = score >= (1.0 - self.threshold)
        logger.debug(
            f"[Histogram] is_color_similar={is_similar} "
            f"(score={score:.4f} >= threshold={1.0 - self.threshold:.4f})"
        )
        return is_similar, score

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _load_bgr(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Load an image as a BGR NumPy array (OpenCV native format).

        Args:
            source: File path (str/Path) or pre-loaded BGR numpy array.

        Returns:
            BGR uint8 array, or None on failure.
        """
        if isinstance(source, np.ndarray):
            # Validate it's a 3-channel colour image
            if source.ndim == 3 and source.shape[2] == 3:
                return source
            elif source.ndim == 2:
                # Greyscale → convert to 3-channel for HSV conversion
                return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            else:
                logger.error(
                    f"[Histogram] Unexpected array shape: {source.shape}"
                )
                return None

        path = Path(source)
        if not path.exists():
            logger.warning(f"[Histogram] File not found: {path}")
            return None

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"[Histogram] OpenCV failed to read: {path}")
            return None

        # Reject tiny images — too little data for meaningful histogram
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            logger.warning(
                f"[Histogram] Image too small ({w}×{h}), skipping: {path}"
            )
            return None

        logger.debug(f"[Histogram] Loaded image: {path} shape={img.shape}")
        return img


# ── Module-level singleton ────────────────────────────────────────────────────
#   from core.histogram import histogram_matcher
#   score = histogram_matcher.compare("img_a.jpg", "img_b.jpg")

histogram_matcher = HistogramMatcher()
