"""
core/sift_matcher.py — SIFT Feature Matching with FLANN
========================================================

What it does:
    Detects and matches local keypoints between two images using:
      • SIFT  (Scale-Invariant Feature Transform) — robust descriptor
      • FLANN (Fast Library for Approximate Nearest Neighbours) — fast search
      • Lowe's Ratio Test — filters ambiguous / noisy matches

Why SIFT?
    Unlike pHash (which compares global structure), SIFT finds specific
    landmark points (corners, edges, blobs) that remain stable under:
      - Scaling and rotation
      - Moderate affine transformations
      - Partial occlusion
      - Lighting changes

Algorithm:
    1. Convert images to greyscale
    2. Detect keypoints + compute 128-dim descriptors via SIFT
    3. Use FLANN KNN (k=2) to find the two nearest descriptor matches
    4. Apply Lowe's ratio test: keep match only if
           dist_to_nearest < SIFT_RATIO_TEST_THRESHOLD × dist_to_2nd_nearest
    5. Normalise score: good_matches / max(kp1, kp2) → [0, 1]

Usage:
    from core.sift_matcher import SIFTMatcher
    matcher = SIFTMatcher()
    result  = matcher.match("img_a.jpg", "img_b.jpg")
    print(result.score, result.good_matches, result.total_keypoints)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings, SIFT_RATIO_TEST_THRESHOLD


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SIFTResult:
    """
    Holds the output of a single SIFT comparison.

    Attributes:
        score            : Normalised similarity in [0.0, 1.0].
                           Formula: good_matches / max(kp_a, kp_b)
        confidence_score : ML-style confidence with 0.8× scaling.
                           Formula: min(1.0, good_matches / (max_kp × 0.8))
                           This gently boosts strong matches while keeping
                           weak matches low — looks more natural on charts.
        good_matches     : Matches that passed Lowe's ratio test.
        total_keypoints  : max(keypoints_img1, keypoints_img2) — denominator.
        keypoints_a      : Raw keypoints detected in image A.
        keypoints_b      : Raw keypoints detected in image B.
        matches          : All raw DMatch objects before ratio filtering.
        good_match_list  : DMatch objects that passed ratio test.
        error            : Error message if matching failed, else None.
    """
    score: float = 0.0
    confidence_score: float = 0.0      # ← confidence-scaled (0.8× boost)
    good_matches: int = 0
    total_keypoints: int = 0
    keypoints_a: List = field(default_factory=list)
    keypoints_b: List = field(default_factory=list)
    matches: List = field(default_factory=list)
    good_match_list: List = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_match(self) -> bool:
        """True if good_matches exceeds the configured minimum."""
        return self.good_matches >= settings.SIFT_MIN_GOOD_MATCHES

    def summary(self) -> str:
        return (
            f"SIFTResult(score={self.score:.4f}, confidence={self.confidence_score:.4f}, "
            f"good={self.good_matches}, total_kp={self.total_keypoints}, "
            f"is_match={self.is_match})"
        )


# ── Core Matcher ──────────────────────────────────────────────────────────────

class SIFTMatcher:
    """
    Stateful SIFT + FLANN matcher.

    The SIFT detector and FLANN index are created once at __init__ and reused
    across multiple match() calls for efficiency.

    Args:
        max_features (int):  Max keypoints to detect per image.
        lowe_ratio   (float): Lowe's ratio test threshold.
        min_matches  (int):   Minimum good matches to consider a pair similar.
    """

    def __init__(
        self,
        max_features: int = settings.SIFT_MAX_FEATURES,
        lowe_ratio: float = SIFT_RATIO_TEST_THRESHOLD,     # from FORMULA CONSTANTS
        min_matches: int = settings.SIFT_MIN_GOOD_MATCHES,
    ) -> None:
        self.max_features = max_features
        self.lowe_ratio = lowe_ratio
        self.min_matches = min_matches

        # ── Create SIFT Detector ──────────────────────────────────────────────
        # nfeatures   : cap number of keypoints (performance control)
        # nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6
        # are the original Lowe defaults — best general-purpose settings.
        self._sift = cv2.SIFT_create(nfeatures=self.max_features)

        # ── Build FLANN Matcher ───────────────────────────────────────────────
        # FLANN_INDEX_KDTREE=1 → KD-tree index, optimal for float descriptors
        # trees=5 means 5 parallel KD-trees for better accuracy vs speed trade-off
        FLANN_INDEX_KDTREE = 1
        index_params  = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        # checks=50: number of tree nodes to traverse per query — higher = slower but more accurate
        search_params = {"checks": 50}
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

        logger.debug(
            f"[SIFT] Initialised | max_features={max_features} "
            f"lowe_ratio={lowe_ratio} min_matches={min_matches}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def match(
        self,
        source_a: Union[str, Path, np.ndarray],
        source_b: Union[str, Path, np.ndarray],
    ) -> SIFTResult:
        """
        Full SIFT matching pipeline between two images.

        Args:
            source_a: Image A — file path or BGR NumPy array (from OpenCV).
            source_b: Image B — file path or BGR NumPy array (from OpenCV).

        Returns:
            SIFTResult with score, match count, and diagnostic data.
        """
        # ── Step 1: Load ──────────────────────────────────────────────────────
        img_a = self._load_grey(source_a)
        img_b = self._load_grey(source_b)

        if img_a is None:
            return SIFTResult(error=f"Failed to load image A: {source_a}")
        if img_b is None:
            return SIFTResult(error=f"Failed to load image B: {source_b}")

        # ── Step 2: Detect keypoints + compute descriptors ────────────────────
        kp_a, des_a = self._detect_and_describe(img_a, label="A")
        kp_b, des_b = self._detect_and_describe(img_b, label="B")

        if des_a is None or des_b is None:
            return SIFTResult(
                keypoints_a=kp_a or [],
                keypoints_b=kp_b or [],
                error="Descriptor computation failed (no keypoints found)"
            )

        if len(des_a) < 2 or len(des_b) < 2:
            return SIFTResult(
                keypoints_a=kp_a,
                keypoints_b=kp_b,
                error=(
                    f"Insufficient descriptors for KNN matching "
                    f"(A={len(des_a)}, B={len(des_b)}) — need ≥ 2 each"
                )
            )

        # ── Step 3: KNN matching (k=2 nearest neighbours) ────────────────────
        # knnMatch returns pairs of matches for each descriptor in des_a.
        # We need k=2 so we can apply Lowe's ratio test.
        try:
            raw_matches = self._flann.knnMatch(des_a, des_b, k=2)
        except cv2.error as exc:
            return SIFTResult(
                keypoints_a=kp_a,
                keypoints_b=kp_b,
                error=f"FLANN matching failed: {exc}"
            )

        # ── Step 4: Lowe's Ratio Test ─────────────────────────────────────────
        # For each pair (best_match, second_best_match):
        #   keep only if distance_to_best < lowe_ratio × distance_to_second_best
        # This discards ambiguous matches where two keypoints are roughly
        # equidistant in descriptor space → unreliable correspondence.
        good_matches = self._apply_ratio_test(raw_matches)

        # ── Step 5: Compute normalised score ──────────────────────────────────
        # Denominator = max(kp_a, kp_b) so the score normalises properly
        # across images of different sizes and complexity.
        total_kp = max(len(kp_a), len(kp_b))
        score, confidence = self._compute_score(len(good_matches), total_kp)

        result = SIFTResult(
            score=score,
            confidence_score=confidence,
            good_matches=len(good_matches),
            total_keypoints=total_kp,
            keypoints_a=kp_a,
            keypoints_b=kp_b,
            matches=raw_matches,
            good_match_list=good_matches,
        )

        logger.debug(f"[SIFT] {result.summary()}")
        return result

    def match_arrays(
        self,
        array_a: np.ndarray,
        array_b: np.ndarray,
    ) -> SIFTResult:
        """
        Convenience wrapper for matching two OpenCV BGR numpy arrays directly
        (e.g. video frames extracted by video.py).
        """
        return self.match(array_a, array_b)

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _load_grey(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Load an image as an 8-bit greyscale numpy array.

        SIFT operates on single-channel images.  We convert here so the
        rest of the pipeline never has to worry about channel count.

        Args:
            source: File path or BGR numpy array.

        Returns:
            2D uint8 numpy array (greyscale), or None on failure.
        """
        if isinstance(source, np.ndarray):
            if source.ndim == 3:
                return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            elif source.ndim == 2:
                return source
            else:
                logger.error(f"[SIFT] Unexpected array shape: {source.shape}")
                return None

        path = Path(source)
        if not path.exists():
            logger.warning(f"[SIFT] File not found: {path}")
            return None

        # cv2.IMREAD_GRAYSCALE loads directly to greyscale — skips BGR decode
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"[SIFT] OpenCV failed to read: {path}")
            return None

        logger.debug(f"[SIFT] Loaded greyscale image: {path} shape={img.shape}")
        return img

    def _detect_and_describe(
        self,
        grey_img: np.ndarray,
        label: str = "?",
    ) -> Tuple[List, Optional[np.ndarray]]:
        """
        Run SIFT detection and descriptor computation on a greyscale image.

        Returns:
            (keypoints, descriptors)
            descriptors is a float32 (N×128) array, or None if none found.
        """
        try:
            keypoints, descriptors = self._sift.detectAndCompute(grey_img, mask=None)
        except cv2.error as exc:
            logger.error(f"[SIFT] detectAndCompute failed on image {label}: {exc}")
            return [], None

        if descriptors is None or len(keypoints) == 0:
            logger.warning(f"[SIFT] No keypoints found in image {label}")
            return [], None

        # FLANN with KD-tree requires float32 descriptors
        descriptors = descriptors.astype(np.float32)

        logger.debug(
            f"[SIFT] Image {label}: detected {len(keypoints)} keypoints, "
            f"descriptors shape={descriptors.shape}"
        )
        return list(keypoints), descriptors

    def _apply_ratio_test(
        self,
        raw_matches: List,
    ) -> List:
        """
        Filter raw KNN matches using Lowe's ratio test.

        A match m is kept when:
            m.distance < self.lowe_ratio × n.distance
        where m is the nearest and n is the second-nearest descriptor match.

        Args:
            raw_matches: Output of FlannBasedMatcher.knnMatch(k=2).

        Returns:
            List of good DMatch objects.
        """
        good = []
        for pair in raw_matches:
            # knnMatch may return fewer than k=2 results for edge descriptors
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)

        logger.debug(
            f"[SIFT] Ratio test: {len(raw_matches)} raw → {len(good)} good matches"
        )
        return good

    def _compute_score(
        self, good_count: int, total_keypoints: int
    ) -> Tuple[float, float]:
        """
        Compute raw and confidence-scaled SIFT scores.

        Raw score:
            score = good_matches / max(keypoints_img1, keypoints_img2)
            This normalises across different image sizes and prevents
            inflated scores when one image has very few keypoints.

        Confidence score (ML-style boost):
            confidence = min(1.0, good_matches / (max_keypoints × 0.8))
            The 0.8 factor means a pair only needs 80% of keypoints
            matched to hit confidence=1.0.  This:
              - Gently boosts strong matches (looks more natural)
              - Keeps weak matches low (no false confidence)
              - Produces smoother score distributions in dashboards

        Args:
            good_count      : Matches that passed Lowe's ratio test.
            total_keypoints : max(kp_img1, kp_img2).

        Returns:
            (raw_score, confidence_score), each in [0.0, 1.0].
        """
        if total_keypoints == 0:
            return 0.0, 0.0

        # Raw normalised score
        raw_score = good_count / total_keypoints
        raw_score = max(0.0, min(1.0, raw_score))

        # Confidence-scaled score — uses 80% ceiling for gentle boost
        # Example: 160 good / 200 total → raw=0.80, confidence=min(1.0, 160/160)=1.0
        confidence = min(1.0, good_count / (total_keypoints * 0.8))
        confidence = max(0.0, confidence)

        logger.debug(
            f"[SIFT] _compute_score: good={good_count} total_kp={total_keypoints} "
            f"raw={raw_score:.4f} confidence={confidence:.4f}"
        )
        return raw_score, confidence


# ── Module-level singleton ────────────────────────────────────────────────────
# Import and use directly:
#   from core.sift_matcher import sift_matcher
#   result = sift_matcher.match("query.jpg", "corpus_item.jpg")

sift_matcher = SIFTMatcher()
