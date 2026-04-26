"""
core/fingerprint.py — Perceptual Hash Fingerprinting
=====================================================

What it does:
    Computes a perceptual hash (pHash) for an image and measures the
    Hamming distance between two hashes.  Unlike cryptographic hashes,
    perceptual hashes are designed so that *visually similar* images
    produce *similar* hashes — small structural edits (resize, mild
    compression, slight colour shift) barely change the hash.

Algorithm:
    1. Resize image to hash_size × hash_size (default 16×16)
    2. Convert to greyscale
    3. Apply DCT (Discrete Cosine Transform) across rows and columns
    4. Retain the top-left 8×8 block (low-frequency components)
    5. Set each bit: 1 if pixel > mean, 0 otherwise → 64-bit fingerprint

Similarity score:
    score = 1 − (hamming_distance / PHASH_MAX_BITS)
    Range: 0.0 (completely different) → 1.0 (identical)

Usage:
    from core.fingerprint import ImageFingerprinter
    fp = ImageFingerprinter()
    hash_a = fp.compute_hash("image_a.jpg")
    hash_b = fp.compute_hash("image_b.jpg")
    score  = fp.similarity(hash_a, hash_b)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError
from loguru import logger

# Import config — works whether run from ai/ root or inside a sub-package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings, PHASH_MAX_BITS


# ── Type alias ────────────────────────────────────────────────────────────────
# imagehash.ImageHash is the object returned by imagehash.phash() etc.
HashType = imagehash.ImageHash


class ImageFingerprinter:
    """
    Computes and compares perceptual hashes for images.

    Args:
        hash_size (int):  Grid dimension for the DCT hash.
                          16 → 256-bit fingerprint.
                          Higher = more precise but slower.
        threshold (int):  Maximum Hamming distance to call two images
                          'potentially similar'.  Used in is_candidate().
    """

    def __init__(
        self,
        hash_size: int = settings.PHASH_HASH_SIZE,
        threshold: int = settings.PHASH_THRESHOLD,
    ) -> None:
        self.hash_size = hash_size
        self.threshold = threshold
        logger.debug(
            f"[Fingerprinter] Initialised | hash_size={hash_size} "
            f"threshold={threshold} max_bits={PHASH_MAX_BITS}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_hash(
        self,
        source: Union[str, Path, Image.Image],
    ) -> Optional[HashType]:
        """
        Compute the perceptual hash of an image.

        Args:
            source: File path (str or Path) OR a pre-loaded PIL Image.

        Returns:
            An imagehash.ImageHash object, or None on failure.
        """
        image = self._load_image(source)
        if image is None:
            return None

        try:
            # imagehash.phash uses DCT-based perceptual hashing
            # hash_size controls the grid; highfreq_factor controls DCT size
            h = imagehash.phash(image, hash_size=self.hash_size)
            logger.debug(f"[Fingerprinter] Computed hash={h}")
            return h
        except Exception as exc:
            logger.error(f"[Fingerprinter] Hash computation failed: {exc}")
            return None

    def hamming_distance(
        self,
        hash_a: HashType,
        hash_b: HashType,
    ) -> int:
        """
        Compute the Hamming distance between two perceptual hashes.

        Hamming distance counts the number of bit positions that differ.
        Range: 0 (identical) → PHASH_MAX_BITS (completely different).

        Args:
            hash_a: Hash of the first image.
            hash_b: Hash of the second image.

        Returns:
            Integer Hamming distance in [0, PHASH_MAX_BITS].
        """
        distance = hash_a - hash_b   # imagehash overloads __sub__ as Hamming
        logger.debug(f"[Fingerprinter] Hamming distance={distance}")
        return distance

    def similarity_score(
        self,
        hash_a: HashType,
        hash_b: HashType,
    ) -> float:
        """
        Convert Hamming distance to a normalised similarity score in [0, 1].

        Formula:
            score = 1 - (hamming_distance / PHASH_MAX_BITS)

        Examples:
            distance=0   → score=1.00  (perfect match)
            distance=32  → score=0.50  (50% similar)
            distance=64  → score=0.00  (completely different)

        Args:
            hash_a: Hash of the first image.
            hash_b: Hash of the second image.

        Returns:
            Float in [0.0, 1.0].
        """
        distance = self.hamming_distance(hash_a, hash_b)
        score = 1.0 - (distance / PHASH_MAX_BITS)
        # Clamp to [0, 1] to handle edge cases from non-standard hash sizes
        score = max(0.0, min(1.0, score))
        logger.debug(f"[Fingerprinter] Similarity score={score:.4f}")
        return score

    # ── Soft Early-Exit Thresholds ──────────────────────────────────────────
    # pHash is *unreliable* for rotated, cropped, and perspective-changed images.
    # A hard `if not is_candidate: skip` would miss real matches.
    # Instead we use a 3-tier soft gate:
    #
    #   STRONG  (score ≥ threshold_score) → definitely run full pipeline
    #   WEAK    (0.2 ≤ score < threshold) → deprioritised but still analysed
    #   SKIP    (score < 0.2)             → only skipped if ENABLE_EARLY_EXIT
    #
    # The 0.2 floor means only *truly hopeless* pairs are discarded.

    EARLY_EXIT_HARD_FLOOR: float = 0.2   # below this, skip is allowed

    def soft_gate(
        self,
        hash_a: HashType,
        hash_b: HashType,
    ) -> str:
        """
        Soft pre-filter using pHash similarity.

        Unlike a hard boolean gate, this returns a priority level so the
        matcher can make smarter decisions:

            'STRONG' — high pHash similarity → run full pipeline with confidence
            'WEAK'   — moderate similarity   → still run SIFT (catches rotation/crop)
            'SKIP'   — very low similarity   → safe to skip (only with ENABLE_EARLY_EXIT)

        Why soft?
            pHash fails for rotated / cropped / perspective-changed images.
            A hard gate would incorrectly discard those real matches.
            Soft gating deprioritises them but still allows SIFT to catch them.

        Args:
            hash_a: Hash of the query image.
            hash_b: Hash of the corpus image.

        Returns:
            'STRONG', 'WEAK', or 'SKIP'.
        """
        score = self.similarity_score(hash_a, hash_b)

        if score >= (1.0 - self.threshold / PHASH_MAX_BITS):
            priority = "STRONG"
        elif score >= self.EARLY_EXIT_HARD_FLOOR:
            priority = "WEAK"
        else:
            priority = "SKIP"

        logger.debug(
            f"[Fingerprinter] soft_gate={priority} "
            f"(score={score:.4f}, threshold_score="
            f"{1.0 - self.threshold / PHASH_MAX_BITS:.4f}, "
            f"hard_floor={self.EARLY_EXIT_HARD_FLOOR})"
        )
        return priority

    def is_candidate(
        self,
        hash_a: HashType,
        hash_b: HashType,
    ) -> bool:
        """
        Backward-compatible gate — wraps soft_gate().

        Returns True for STRONG *and* WEAK priorities, so only truly
        hopeless pairs (SKIP) are rejected. This ensures rotated / cropped
        images are never silently discarded.
        """
        return self.soft_gate(hash_a, hash_b) != "SKIP"

    def hash_from_array(self, array: np.ndarray) -> Optional[HashType]:
        """
        Compute a perceptual hash directly from a NumPy array (e.g. a video frame).

        Args:
            array: uint8 NumPy array, shape (H, W) or (H, W, C).

        Returns:
            imagehash.ImageHash or None on failure.
        """
        try:
            # Convert BGR (OpenCV) → RGB (PIL) if it's a 3-channel image
            if array.ndim == 3 and array.shape[2] == 3:
                array = array[:, :, ::-1]   # BGR → RGB

            image = Image.fromarray(array.astype(np.uint8))
            return self.compute_hash(image)
        except Exception as exc:
            logger.error(f"[Fingerprinter] hash_from_array failed: {exc}")
            return None

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _load_image(
        self,
        source: Union[str, Path, Image.Image],
    ) -> Optional[Image.Image]:
        """
        Load an image from a file path or pass-through a PIL Image.

        Converts to RGB to ensure consistent channel handling across
        JPEG, PNG, GIF, WebP, etc.

        Args:
            source: File path or PIL Image object.

        Returns:
            PIL Image in RGB mode, or None on error.
        """
        if isinstance(source, Image.Image):
            # Already loaded — just normalise to RGB
            return source.convert("RGB")

        path = Path(source)

        if not path.exists():
            logger.warning(f"[Fingerprinter] File not found: {path}")
            return None

        if not path.is_file():
            logger.warning(f"[Fingerprinter] Path is not a file: {path}")
            return None

        try:
            image = Image.open(path).convert("RGB")
            logger.debug(f"[Fingerprinter] Loaded image: {path} size={image.size}")
            return image
        except UnidentifiedImageError:
            logger.error(f"[Fingerprinter] Unrecognised image format: {path}")
            return None
        except Exception as exc:
            logger.error(f"[Fingerprinter] Failed to load image {path}: {exc}")
            return None


# ── Module-level singleton ────────────────────────────────────────────────────
# Most code should import this pre-built instance rather than instantiating
# their own, to avoid redundant configuration reads.
#
#   from core.fingerprint import fingerprinter
#   score = fingerprinter.similarity_score(hash_a, hash_b)

fingerprinter = ImageFingerprinter()
