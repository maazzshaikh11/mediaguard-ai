"""
config.py — Central Configuration for AI Media Matching System
==============================================================

All tuneable parameters live here:
  - Matching thresholds
  - Algorithm weights for the composite scorer
  - File paths
  - Supported media formats
  - Logging configuration

Usage:
    from config import settings
    print(settings.PHASH_THRESHOLD)

All values can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()
from pydantic import Field
from typing import List


# ── Base Directories ──────────────────────────────────────────────────────────

# Root of the `ai/` package (directory containing this file)
BASE_DIR: Path = Path(__file__).resolve().parent

# Corpus directory — stores reference media files that incoming content is
# compared against. Can be overridden via CORPUS_DIR env var.
DEFAULT_CORPUS_DIR: Path = BASE_DIR / "corpus"

# Temporary upload directory — holds files during request processing
DEFAULT_UPLOAD_DIR: Path = BASE_DIR / "uploads"


# ===================== FORMULA CONSTANTS =====================
# These are used directly inside algorithm math — imported as module-level
# constants so individual modules don't need to instantiate Settings.
# They are ALSO exposed as Pydantic fields below so they can be overridden
# via environment variables if needed.

# pHash: a 16×16 perceptual hash produces a 256-bit (not 64-bit) fingerprint,
# but the *Hamming distance* range for comparison is always 0–64 bits per chunk.
# We normalise similarity as:  score = 1 - (hamming_distance / PHASH_MAX_BITS)
PHASH_MAX_BITS: int = 64          # Max Hamming distance for normalisation

# SIFT: Lowe's ratio test keeps a match only when
#   distance_to_nearest / distance_to_2nd_nearest  <  SIFT_RATIO_TEST_THRESHOLD
# This filters ambiguous / low-confidence keypoint correspondences.
SIFT_RATIO_TEST_THRESHOLD: float = 0.75

# Histogram: raw Chi-square distances are converted to a [0,1] similarity via
# exponential decay:  score = exp(−chi2 / HISTOGRAM_NORMALIZATION_FACTOR)
# Note: since histograms are L1-normalized to sum to 1.0, max chi2 is ~1.0.
# A factor of 0.5 means disjoint histograms score exp(-2) ≈ 0.13.
HISTOGRAM_NORMALIZATION_FACTOR: float = 0.5

# OCR: tokens shorter than this are discarded before Jaccard computation
# to avoid noise words like 'a', 'to', 'of' skewing the score.
OCR_MIN_TOKEN_LENGTH: int = 3

# Video: fraction of the SIFT/histogram composite score that a single frame pair
# must exceed to count as a "matched frame" in the temporal similarity sweep.
VIDEO_FRAME_MATCH_THRESHOLD: float = 0.5

# ── Histogram Clamping ──────────────────────────────────────────────────────
# Histogram measures colour distribution, NOT object identity.
# Hard cap prevents histogram from dominating the composite score.
HISTOGRAM_MAX_SIMILARITY: float = 0.85

# When SIFT structural score is below this, halve histogram influence
# (same colours ≠ same content)
HISTOGRAM_SIFT_DAMPENING_THRESHOLD: float = 0.2

# ── Signal Reliability Thresholds ───────────────────────────────────────────
# Minimum OCR tokens on EACH side for OCR to be considered reliable
RELIABILITY_OCR_MIN_TOKENS: int = 3

# Minimum image area (h × w) for histogram to be statistically meaningful
RELIABILITY_MIN_IMAGE_AREA: int = 10_000  # ~100×100

# SIFT keypoints below this → unreliable structural comparison
RELIABILITY_MIN_SIFT_KEYPOINTS: int = 15

# ── Video Optimization ──────────────────────────────────────────────────────
# Frame-level early exit: skip remaining candidate frames if score exceeds this
VIDEO_EARLY_EXIT_SCORE: float = 0.85

# Max seconds for FFmpeg fallback extraction
VIDEO_FFMPEG_TIMEOUT: int = 30

# Temporal window: compare frame_a[i] only against frame_b[i ± window]
VIDEO_TEMPORAL_WINDOW: int = 3

# Standardised frame size for consistent keypoint detection
VIDEO_FRAME_WIDTH: int = 640
VIDEO_FRAME_HEIGHT: int = 480

# Frame-level scoring weights (must sum to 1.0)
VIDEO_SIFT_WEIGHT: float = 0.7
VIDEO_HIST_WEIGHT: float = 0.3

# Temporal consistency bonus: reward when matched frame is temporally aligned
VIDEO_TEMPORAL_BONUS: float = 0.05


# ===================== SYSTEM FLAGS =====================
# Module-level booleans for fast feature-flag checks.
# Mirrored as Pydantic fields so they can be set in .env or environment.

ENABLE_EARLY_EXIT: bool = True    # Skip heavy computation if pHash score is low
ENABLE_LOGGING: bool = True       # Enable loguru structured logging
DEBUG_MODE: bool = True           # Verbose debug output (set False in production)


# ── Settings Class ────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Central settings object.
    Values are read in priority order:
      1. Environment variables
      2. .env file (if present)
      3. Defaults defined below
    """

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "MediaGuard AI Matching Engine"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # ── System Flags (overrideable via env / .env) ────────────────────────────
    DEBUG_MODE: bool = Field(
        default=True,
        description="Verbose debug output — set to False in production"
    )
    ENABLE_EARLY_EXIT: bool = Field(
        default=True,
        description=(
            "When True, the matcher skips SIFT/Histogram/OCR if the pHash "
            "score is too low (below EARLY_EXIT_PHASH_SCORE), saving CPU time."
        )
    )
    ENABLE_LOGGING: bool = Field(
        default=True,
        description="Toggle loguru structured logging on/off"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    CORPUS_DIR: Path = Field(
        default=DEFAULT_CORPUS_DIR,
        description="Directory containing reference media corpus"
    )
    UPLOAD_DIR: Path = Field(
        default=DEFAULT_UPLOAD_DIR,
        description="Temp directory for uploaded files during processing"
    )

    # ── Supported Formats ─────────────────────────────────────────────────────
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"],
        description="Accepted image file extensions"
    )
    SUPPORTED_VIDEO_FORMATS: List[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"],
        description="Accepted video file extensions"
    )

    # ── Perceptual Hashing (fingerprint.py) ───────────────────────────────────
    PHASH_THRESHOLD: int = Field(
        default=10,
        description=(
            "Maximum Hamming distance between two pHash values to be "
            "considered a potential match. "
            "Range: 0 (identical) – 64 (completely different). "
            "Lower = stricter."
        )
    )
    PHASH_HASH_SIZE: int = Field(
        default=16,
        description="Hash grid size. 16 → 256-bit hash (good balance)."
    )

    # ── SIFT Feature Matching (sift_matcher.py) ───────────────────────────────
    SIFT_MIN_GOOD_MATCHES: int = Field(
        default=10,
        description="Minimum number of Lowe-filtered SIFT matches to consider pair similar"
    )
    SIFT_LOWE_RATIO: float = Field(
        default=0.75,
        description=(
            "Lowe's ratio test threshold. "
            "A match is kept if nearest_distance / second_nearest_distance < ratio. "
            "Lower = stricter (fewer but more reliable matches)."
        )
    )
    SIFT_MAX_FEATURES: int = Field(
        default=2000,
        description="Maximum keypoints extracted per image by SIFT"
    )

    # ── Histogram Matching (histogram.py) ─────────────────────────────────────
    HIST_THRESHOLD: float = Field(
        default=0.3,
        description=(
            "Maximum Chi-square distance between HSV histograms "
            "to consider images color-similar. "
            "Lower = stricter."
        )
    )
    HIST_BINS: int = Field(
        default=50,
        description="Number of bins per HSV channel in the histogram"
    )

    # ── OCR Text Matching (ocr.py) ────────────────────────────────────────────
    OCR_MIN_TEXT_LENGTH: int = Field(
        default=10,
        description="Minimum character count for OCR text to be used in matching"
    )
    OCR_JACCARD_THRESHOLD: float = Field(
        default=0.3,
        description=(
            "Minimum Jaccard similarity between OCR token sets "
            "to consider text similar. "
            "Range: 0.0 – 1.0. Higher = stricter."
        )
    )
    OCR_LANG: str = Field(
        default="eng",
        description="Tesseract language code(s), e.g. 'eng+hin' for multi-language"
    )

    # ── Video Processing (video.py) ───────────────────────────────────────────
    VIDEO_MAX_FRAMES: int = Field(
        default=30,
        description="Maximum number of frames sampled per video for comparison"
    )
    VIDEO_FRAME_INTERVAL: int = Field(
        default=30,
        description="Sample one frame every N frames (e.g. 30 → ~1fps for 30fps video)"
    )
    VIDEO_SIMILARITY_THRESHOLD: float = Field(
        default=0.6,
        description=(
            "Fraction of matched frames required to call a video pair similar. "
            "e.g. 0.6 = at least 60% of sampled frames must match."
        )
    )
    VIDEO_FRAME_MATCH_THRESHOLD: float = Field(
        default=VIDEO_FRAME_MATCH_THRESHOLD,   # mirrors module-level constant
        description=(
            "Minimum composite score a single frame pair must achieve to count "
            "as a 'matched' frame during temporal sweep. Range: 0.0 – 1.0."
        )
    )
    VIDEO_EARLY_EXIT_SCORE: float = Field(
        default=VIDEO_EARLY_EXIT_SCORE,
        description="Skip remaining candidate frames if match score exceeds this"
    )
    VIDEO_FFMPEG_TIMEOUT: int = Field(
        default=VIDEO_FFMPEG_TIMEOUT,
        description="Max seconds for FFmpeg fallback frame extraction"
    )
    VIDEO_TEMPORAL_WINDOW: int = Field(
        default=VIDEO_TEMPORAL_WINDOW,
        description="Compare frame_a[i] only against frame_b[i ± window]"
    )
    VIDEO_FRAME_WIDTH: int = Field(
        default=VIDEO_FRAME_WIDTH,
        description="Standardised frame width for preprocessing"
    )
    VIDEO_FRAME_HEIGHT: int = Field(
        default=VIDEO_FRAME_HEIGHT,
        description="Standardised frame height for preprocessing"
    )
    VIDEO_SIFT_WEIGHT: float = Field(
        default=VIDEO_SIFT_WEIGHT,
        description="SIFT weight in frame-level composite score (structure)"
    )
    VIDEO_HIST_WEIGHT: float = Field(
        default=VIDEO_HIST_WEIGHT,
        description="Histogram weight in frame-level composite score (colour)"
    )
    VIDEO_TEMPORAL_BONUS: float = Field(
        default=VIDEO_TEMPORAL_BONUS,
        description="Bonus added when matched frame is temporally aligned"
    )

    # ── Composite Scorer Weights (scorer.py) ──────────────────────────────────
    # Must sum to 1.0 (enforced at runtime)
    WEIGHT_PHASH: float = Field(
        default=0.30,
        description="Weight for perceptual hash score in composite calculation"
    )
    WEIGHT_SIFT: float = Field(
        default=0.35,
        description="Weight for SIFT feature match score"
    )
    WEIGHT_HISTOGRAM: float = Field(
        default=0.20,
        description="Weight for HSV histogram similarity score"
    )
    WEIGHT_OCR: float = Field(
        default=0.15,
        description="Weight for OCR text Jaccard similarity score"
    )

    # ── Early Exit Optimization (matcher.py) ──────────────────────────────────
    EARLY_EXIT_PHASH_SCORE: float = Field(
        default=0.95,
        description=(
            "If pHash score exceeds this threshold, skip SIFT/Histogram/OCR "
            "and return the result immediately (speed optimization)."
        )
    )
    EARLY_EXIT_MIN_COMPOSITE: float = Field(
        default=0.85,
        description=(
            "If composite score exceeds this, stop processing remaining corpus items."
        )
    )

    # ── Formula Constants (Pydantic-exposed for env override) ─────────────────
    PHASH_MAX_BITS: int = Field(
        default=PHASH_MAX_BITS,
        description="Max Hamming distance used to normalise pHash similarity to [0,1]"
    )
    SIFT_RATIO_TEST_THRESHOLD: float = Field(
        default=SIFT_RATIO_TEST_THRESHOLD,
        description="Lowe's ratio test threshold for SIFT match filtering"
    )
    HISTOGRAM_NORMALIZATION_FACTOR: float = Field(
        default=HISTOGRAM_NORMALIZATION_FACTOR,
        description="Exponential decay denominator for chi-square → similarity conversion"
    )
    OCR_MIN_TOKEN_LENGTH: int = Field(
        default=OCR_MIN_TOKEN_LENGTH,
        description="Discard OCR tokens shorter than this before Jaccard computation"
    )
    HISTOGRAM_MAX_SIMILARITY: float = Field(
        default=HISTOGRAM_MAX_SIMILARITY,
        description="Hard cap on histogram similarity — prevents false-positive dominance"
    )
    HISTOGRAM_SIFT_DAMPENING_THRESHOLD: float = Field(
        default=HISTOGRAM_SIFT_DAMPENING_THRESHOLD,
        description="Below this SIFT score, histogram similarity is halved"
    )

    # ── Signal Reliability (Pydantic-exposed) ─────────────────────────────────
    RELIABILITY_OCR_MIN_TOKENS: int = Field(
        default=RELIABILITY_OCR_MIN_TOKENS,
        description="Min OCR tokens per side for OCR signal to be reliable"
    )
    RELIABILITY_MIN_IMAGE_AREA: int = Field(
        default=RELIABILITY_MIN_IMAGE_AREA,
        description="Min image area (h×w) for histogram to be meaningful"
    )
    RELIABILITY_MIN_SIFT_KEYPOINTS: int = Field(
        default=RELIABILITY_MIN_SIFT_KEYPOINTS,
        description="Min SIFT keypoints for structural match to be reliable"
    )

    # ── Final Match Decision ──────────────────────────────────────────────────
    MATCH_CONFIDENCE_THRESHOLD: float = Field(
        default=0.60,
        description=(
            "Minimum composite score for a corpus item to be returned as a match. "
            "Results below this threshold are discarded."
        )
    )
    TOP_K_RESULTS: int = Field(
        default=5,
        description="Maximum number of top matches to return per query"
    )

    # ── Tesseract Binary Path ─────────────────────────────────────────────────
    TESSERACT_CMD: str = Field(
        default="/opt/homebrew/bin/tesseract",  # macOS (Homebrew) default
        description=(
            "Absolute path to tesseract binary. "
            "Override if installed elsewhere: "
            "  Linux: /usr/bin/tesseract "
            "  Windows: C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe"
        )
    )

    # ── Pydantic Config ───────────────────────────────────────────────────────
    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",          # Silently ignore unknown env vars
    }

    def validate_weights(self) -> None:
        """
        Ensures that WEIGHT_PHASH + WEIGHT_SIFT + WEIGHT_HISTOGRAM + WEIGHT_OCR == 1.0.
        Called on application startup. Raises ValueError if misconfigured.
        """
        total = (
            self.WEIGHT_PHASH
            + self.WEIGHT_SIFT
            + self.WEIGHT_HISTOGRAM
            + self.WEIGHT_OCR
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scorer weights must sum to 1.0, but got {total:.4f}. "
                f"Check WEIGHT_PHASH, WEIGHT_SIFT, WEIGHT_HISTOGRAM, WEIGHT_OCR."
            )

    def ensure_directories(self) -> None:
        """
        Creates required directories (CORPUS_DIR, UPLOAD_DIR) at startup
        if they don't already exist.
        """
        self.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def is_supported_image(self, filename: str) -> bool:
        """Returns True if the file extension is a supported image format."""
        return Path(filename).suffix.lower() in self.SUPPORTED_IMAGE_FORMATS

    def is_supported_video(self, filename: str) -> bool:
        """Returns True if the file extension is a supported video format."""
        return Path(filename).suffix.lower() in self.SUPPORTED_VIDEO_FORMATS

    def is_supported_media(self, filename: str) -> bool:
        """Returns True if the file is either a supported image or video."""
        return self.is_supported_image(filename) or self.is_supported_video(filename)


# ── Singleton Instance ────────────────────────────────────────────────────────
# Import this wherever config values are needed:
#   from config import settings

settings = Settings()
