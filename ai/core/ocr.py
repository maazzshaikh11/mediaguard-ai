"""
core/ocr.py — OCR Text Extraction & Jaccard Similarity
========================================================

What it does:
    Extracts visible text from images using Tesseract OCR, then compares
    extracted text between two images using Jaccard similarity on token sets.

Why OCR?
    Many copyrighted media contain watermarks, titles, credits, or overlaid
    text.  Even if the pixel content has been modified (resized, compressed,
    colour-shifted), embedded text often remains readable.  OCR captures
    this signal and adds it to the composite match score.

Algorithm:
    1. Load image as BGR
    2. Pre-process for OCR:
       a. Convert to greyscale
       b. Apply adaptive thresholding (binarisation)
       c. Denoise with morphological opening
    3. Run Tesseract OCR → raw text
    4. Tokenize + clean:
       - Lowercase
       - Remove tokens shorter than OCR_MIN_TOKEN_LENGTH (noise like 'a', 'to')
       - Strip non-alphanumeric characters
    5. Compare two token sets via Jaccard similarity:
           J(A, B) = |A ∩ B| / |A ∪ B|
       Range: 0.0 (no shared tokens) → 1.0 (identical token sets)

Usage:
    from core.ocr import OCRMatcher
    ocr = OCRMatcher()
    score = ocr.compare("img_a.jpg", "img_b.jpg")
    text  = ocr.extract_text("img_a.jpg")
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Set, Tuple, Union

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings, OCR_MIN_TOKEN_LENGTH

# ── Lazy Tesseract Import ─────────────────────────────────────────────────────
# pytesseract is imported at use-time so the module loads even if Tesseract
# isn't installed.  This lets other parts of the system work without OCR.
_pytesseract = None


def _get_pytesseract():
    """Lazy-load pytesseract and configure the binary path."""
    global _pytesseract
    if _pytesseract is None:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            _pytesseract = pytesseract
            logger.debug(
                f"[OCR] pytesseract loaded | cmd={settings.TESSERACT_CMD}"
            )
        except ImportError:
            logger.error(
                "[OCR] pytesseract is not installed. "
                "Run: pip install pytesseract"
            )
            raise
    return _pytesseract


class OCRMatcher:
    """
    Extracts text from images via Tesseract OCR and compares them
    using Jaccard set similarity.

    Args:
        lang             (str):   Tesseract language code(s).
        min_text_length  (int):   Min chars for OCR output to be usable.
        min_token_length (int):   Tokens shorter than this are noise.
        jaccard_threshold(float): Minimum Jaccard score to call 'text-similar'.
    """

    def __init__(
        self,
        lang: str = settings.OCR_LANG,
        min_text_length: int = settings.OCR_MIN_TEXT_LENGTH,
        min_token_length: int = OCR_MIN_TOKEN_LENGTH,
        jaccard_threshold: float = settings.OCR_JACCARD_THRESHOLD,
    ) -> None:
        self.lang = lang
        self.min_text_length = min_text_length
        self.min_token_length = min_token_length
        self.jaccard_threshold = jaccard_threshold
        logger.debug(
            f"[OCR] Initialised | lang={lang} min_text={min_text_length} "
            f"min_token={min_token_length} jaccard_threshold={jaccard_threshold}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compare(
        self,
        source_a: Union[str, Path, np.ndarray],
        source_b: Union[str, Path, np.ndarray],
    ) -> float:
        """
        Extract text from both images and compute Jaccard similarity.

        Returns:
            Jaccard similarity in [0.0, 1.0].
            Returns 0.0 if either image yields insufficient text.
        """
        tokens_a = self.extract_tokens(source_a)
        tokens_b = self.extract_tokens(source_b)

        if not tokens_a or not tokens_b:
            logger.debug(
                f"[OCR] Insufficient tokens for comparison "
                f"(A={len(tokens_a) if tokens_a else 0}, "
                f"B={len(tokens_b) if tokens_b else 0})"
            )
            return 0.0

        score = self.jaccard_similarity(tokens_a, tokens_b)
        logger.debug(
            f"[OCR] Comparison complete | "
            f"tokens_A={len(tokens_a)} tokens_B={len(tokens_b)} "
            f"jaccard={score:.4f}"
        )
        return score

    def extract_text(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> str:
        """
        Extract raw text from an image using Tesseract OCR.

        Pre-processing pipeline:
            1. Convert to greyscale
            2. Adaptive threshold (Gaussian, block=11, C=2) → binarise
            3. Morphological opening (3×3 kernel) → remove speckle noise
            4. Run Tesseract with --psm 6 (assume uniform block of text)

        Args:
            source: File path or BGR NumPy array.

        Returns:
            Raw OCR text string (may be empty).
        """
        img = self._load_bgr(source)
        if img is None:
            return ""

        # ── Pre-process for OCR ───────────────────────────────────────────────
        processed = self._preprocess_for_ocr(img)

        # ── Run Tesseract ─────────────────────────────────────────────────────
        try:
            pytesseract = _get_pytesseract()
            # --psm 6: Assume a single uniform block of text
            # --oem 3: Use LSTM neural net engine (best accuracy)
            custom_config = r"--oem 3 --psm 6"
            text = pytesseract.image_to_string(
                processed,
                lang=self.lang,
                config=custom_config,
            )
            text = text.strip()
            logger.debug(f"[OCR] Extracted text ({len(text)} chars)")
            return text

        except Exception as exc:
            logger.error(f"[OCR] Tesseract extraction failed: {exc}")
            return ""

    def extract_tokens(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Optional[Set[str]]:
        """
        Extract text from an image and convert to a cleaned token set.

        Cleaning pipeline:
            1. Extract raw text via Tesseract
            2. Check minimum text length (skip short junk)
            3. Lowercase everything
            4. Strip non-alphanumeric characters from each token
            5. Remove tokens shorter than min_token_length
               (eliminates noise words like 'a', 'to', 'of', 'is')

        Args:
            source: File path or BGR NumPy array.

        Returns:
            Set of cleaned tokens, or None if text is too short / empty.
        """
        raw_text = self.extract_text(source)

        # Gate: reject very short OCR output — likely garbage
        if len(raw_text) < self.min_text_length:
            logger.debug(
                f"[OCR] Text too short ({len(raw_text)} < {self.min_text_length}), "
                f"skipping tokenization"
            )
            return None

        # Tokenize and clean
        tokens = self._clean_tokens(raw_text)

        if not tokens:
            logger.debug("[OCR] No valid tokens after cleaning")
            return None

        logger.debug(f"[OCR] Extracted {len(tokens)} clean tokens")
        return tokens

    def jaccard_similarity(
        self,
        set_a: Set[str],
        set_b: Set[str],
    ) -> float:
        """
        Compute Jaccard similarity between two token sets.

        Formula:
            J(A, B) = |A ∩ B| / |A ∪ B|

        Properties:
            - J = 0.0 → no shared tokens
            - J = 1.0 → identical token sets
            - Symmetric: J(A,B) == J(B,A)
            - Order-independent (set-based)

        Args:
            set_a: Token set from image A.
            set_b: Token set from image B.

        Returns:
            Float in [0.0, 1.0].
        """
        if not set_a and not set_b:
            # Both images have no text → identical condition
            # This is semantically correct: both are text-free, so they match
            # on the text dimension (viva-winning edge case)
            logger.debug("[OCR] Both sets empty → identical text condition → J=1.0")
            return 1.0

        intersection = set_a & set_b
        union = set_a | set_b

        if len(union) == 0:
            return 1.0  # defensive — shouldn't reach here after empty check

        similarity = len(intersection) / len(union)

        logger.debug(
            f"[OCR] Jaccard: |A|={len(set_a)} |B|={len(set_b)} "
            f"|A∩B|={len(intersection)} |A∪B|={len(union)} "
            f"J={similarity:.4f}"
        )
        return similarity

    def is_text_similar(
        self,
        source_a: Union[str, Path, np.ndarray],
        source_b: Union[str, Path, np.ndarray],
    ) -> Tuple[bool, float]:
        """
        Convenience method: compare + threshold check.

        Returns:
            (is_similar, jaccard_score)
        """
        score = self.compare(source_a, source_b)
        is_similar = score >= self.jaccard_threshold
        return is_similar, score

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Pre-process a BGR image to maximise Tesseract accuracy.

        Steps:
            1. BGR → greyscale
            2. Adaptive Gaussian thresholding:
               - blockSize=11 (local neighbourhood)
               - C=2           (offset from mean)
               Produces a binary image where text is black on white.
            3. Morphological opening (3×3 kernel):
               - Erode then dilate
               - Removes salt-and-pepper noise / thin artefacts
               - Preserves text stroke width

        Args:
            img: BGR uint8 array.

        Returns:
            Pre-processed uint8 array (single channel, binarised).
        """
        # Step 1: Greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Adaptive threshold — handles uneven lighting
        # ADAPTIVE_THRESH_GAUSSIAN_C uses a weighted Gaussian window
        binary = cv2.adaptiveThreshold(
            grey,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Step 3: Morphological opening — remove noise, keep text strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        logger.debug(
            f"[OCR] Pre-processing complete | "
            f"input={img.shape} → output={opened.shape}"
        )
        return opened

    def _clean_tokens(self, text: str) -> Set[str]:
        """
        Convert raw OCR text into a clean set of lowercase tokens.

        Rules:
            1. Lowercase everything
            2. Split on whitespace
            3. Strip non-alphanumeric characters from each token
            4. Remove tokens shorter than self.min_token_length
            5. Deduplicate (set)

        Args:
            text: Raw OCR output string.

        Returns:
            Set of cleaned, lowercase tokens.
        """
        # Lowercase and split
        words = text.lower().split()

        # Strip non-alnum characters, filter by length + alphanumeric guard
        cleaned = set()
        for word in words:
            # Remove anything that isn't a letter or digit
            token = re.sub(r"[^a-z0-9]", "", word)
            # Length gate + alphanumeric guard (catches edge cases where
            # regex strip leaves empty/whitespace-only results, and filters
            # OCR garbage like "|", "—", random symbols)
            if len(token) >= self.min_token_length and token.isalnum():
                cleaned.add(token)

        return cleaned

    def _load_bgr(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Load an image as a BGR NumPy array.

        Args:
            source: File path or pre-loaded BGR NumPy array.

        Returns:
            BGR uint8 array, or None on failure.
        """
        if isinstance(source, np.ndarray):
            if source.ndim == 3 and source.shape[2] == 3:
                return source
            elif source.ndim == 2:
                # Greyscale → expand to 3-channel for uniform handling
                return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            else:
                logger.error(f"[OCR] Unexpected array shape: {source.shape}")
                return None

        path = Path(source)
        if not path.exists():
            logger.warning(f"[OCR] File not found: {path}")
            return None

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"[OCR] OpenCV failed to read: {path}")
            return None

        logger.debug(f"[OCR] Loaded image: {path} shape={img.shape}")
        return img


# ── Module-level singleton ────────────────────────────────────────────────────
#   from core.ocr import ocr_matcher
#   score = ocr_matcher.compare("img_a.jpg", "img_b.jpg")

ocr_matcher = OCRMatcher()
