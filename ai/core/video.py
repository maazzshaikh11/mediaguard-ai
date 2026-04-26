"""
core/video.py — Video Frame Extraction & Similarity Matching
=============================================================

Architecture:
    video_a → extract frames (1fps, max 30) → [frame_a_0, frame_a_1, ...]
    video_b → extract frames (1fps, max 30) → [frame_b_0, frame_b_1, ...]
                                                        │
                        For each frame_a[i]:            │
                          search frame_b[i ± window]    ├→ best_score_i
                          (temporal window matching)    │
                                                        │
                    video_similarity = matched_frames / total_frames

Key Design Decisions:
    - Window-based matching (NOT O(n²) full scan)
    - Frame preprocessing: resize to 640×480 for consistent keypoints
    - Frame scoring: SIFT 0.7 + Histogram 0.3 (structure > colour)
    - Temporal consistency bonus for timeline-aligned matches
    - OpenCV primary, FFmpeg fallback for codec issues

Usage:
    from core.video import video_matcher
    result = video_matcher.compare("video_a.mp4", "video_b.mp4")
    print(result.similarity, result.matched_frames)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    settings,
    VIDEO_FRAME_MATCH_THRESHOLD,
    VIDEO_EARLY_EXIT_SCORE,
    VIDEO_FFMPEG_TIMEOUT,
    VIDEO_TEMPORAL_WINDOW,
    VIDEO_SIFT_WEIGHT,
    VIDEO_HIST_WEIGHT,
    VIDEO_TEMPORAL_BONUS,
    HISTOGRAM_SIFT_DAMPENING_THRESHOLD,
)


# ── Result Containers ─────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    fps: float = 0.0
    duration: float = 0.0       # seconds
    frame_count: int = 0
    width: int = 0
    height: int = 0
    codec: str = ""

    def summary(self) -> str:
        return (
            f"VideoMetadata(fps={self.fps:.2f}, duration={self.duration:.1f}s, "
            f"frames={self.frame_count}, {self.width}×{self.height}, codec={self.codec})"
        )


@dataclass
class VideoMatchResult:
    """Complete result of a video-vs-video comparison."""
    similarity: float = 0.0            # matched_frames / total_frames
    matched_frames: int = 0            # frames exceeding threshold
    total_frames_a: int = 0            # frames sampled from video A
    total_frames_b: int = 0            # frames sampled from video B
    per_frame_scores: List[float] = field(default_factory=list)
    metadata_a: Optional[VideoMetadata] = None
    metadata_b: Optional[VideoMetadata] = None
    processing_time: float = 0.0       # seconds
    error: Optional[str] = None

    @property
    def is_match(self) -> bool:
        return self.similarity >= settings.VIDEO_SIMILARITY_THRESHOLD

    def summary(self) -> str:
        return (
            f"VideoMatchResult(similarity={self.similarity:.4f}, "
            f"matched={self.matched_frames}/{self.total_frames_a}, "
            f"is_match={self.is_match}, time={self.processing_time:.2f}s)"
        )


# ── Frame Extraction ──────────────────────────────────────────────────────────

class VideoFrameExtractor:
    """
    Extracts representative frames from video files.

    Strategy:
        1. Detect video FPS via OpenCV
        2. Sample 1 frame per second (round(fps) interval)
        3. Cap at VIDEO_MAX_FRAMES (default 30)
        4. Preprocess: resize to 640×480 for consistent keypoint detection
        5. Fallback to FFmpeg if OpenCV fails (codec issues)
    """

    def __init__(
        self,
        max_frames: int = settings.VIDEO_MAX_FRAMES,
        target_width: int = settings.VIDEO_FRAME_WIDTH,
        target_height: int = settings.VIDEO_FRAME_HEIGHT,
    ) -> None:
        self.max_frames = max_frames
        self.target_size = (target_width, target_height)
        logger.debug(
            f"[Video] Extractor init | max_frames={max_frames} "
            f"target_size={self.target_size}"
        )

    def extract_frames(
        self, video_path: Union[str, Path],
    ) -> Tuple[List[np.ndarray], VideoMetadata]:
        """
        Extract preprocessed frames from a video file.

        Primary: OpenCV VideoCapture (fast, no subprocess).
        Fallback: FFmpeg (handles codecs OpenCV can't).

        Args:
            video_path: Path to the video file.

        Returns:
            (list_of_bgr_frames, metadata)
            Frames are resized to target_size for consistency.
            Returns ([], metadata_with_error) on failure.
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"[Video] File not found: {path}")
            return [], VideoMetadata()

        # Try OpenCV first
        frames, meta = self._extract_with_opencv(str(path))

        # Fallback to FFmpeg if OpenCV returned no frames
        if not frames:
            logger.warning(
                f"[Video] OpenCV extraction failed, trying FFmpeg fallback"
            )
            frames, meta = self._extract_with_ffmpeg(str(path))

        if not frames:
            logger.error(f"[Video] All extraction methods failed for: {path}")
            return [], meta

        logger.info(
            f"[Video] Extracted {len(frames)} frames from {path.name} | "
            f"{meta.summary()}"
        )
        return frames, meta

    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract metadata without extracting frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoMetadata()
        meta = self._read_metadata(cap)
        cap.release()
        return meta

    # ── OpenCV Extraction (Primary) ───────────────────────────────────────────

    def _extract_with_opencv(
        self, video_path: str,
    ) -> Tuple[List[np.ndarray], VideoMetadata]:
        """
        Extract frames using OpenCV VideoCapture.

        Sampling logic:
            - sample_interval = round(fps) → 1 frame per second
            - Cap total samples at max_frames
            - Preprocess each frame (resize to target_size)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[Video] OpenCV cannot open: {video_path}")
            return [], VideoMetadata()

        meta = self._read_metadata(cap)

        # Calculate sampling interval — 1 frame per second
        # round() handles 29.97fps → 30, avoiding timing drift
        sample_interval = max(1, round(meta.fps)) if meta.fps > 0 else 1

        frames: List[np.ndarray] = []
        frame_idx = 0

        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # Preprocess: resize for consistent keypoint detection
                processed = self._preprocess_frame(frame)
                if processed is not None:
                    frames.append(processed)

            frame_idx += 1

        cap.release()
        logger.debug(
            f"[Video] OpenCV: read {frame_idx} raw frames, "
            f"sampled {len(frames)} (interval={sample_interval})"
        )
        return frames, meta

    # ── FFmpeg Extraction (Fallback) ──────────────────────────────────────────

    def _extract_with_ffmpeg(
        self, video_path: str,
    ) -> Tuple[List[np.ndarray], VideoMetadata]:
        """
        Fallback frame extraction using FFmpeg subprocess.

        Extracts at 1fps, reads raw RGB frames from stdout pipe.
        Used when OpenCV can't handle the codec.
        """
        meta = self.get_video_metadata(video_path)

        try:
            # FFmpeg command: extract at 1fps, output raw RGB frames
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1,scale={self.target_size[0]}:{self.target_size[1]}",
                "-frames:v", str(self.max_frames),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-loglevel", "error",
                "-"  # pipe to stdout
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=settings.VIDEO_FFMPEG_TIMEOUT,
            )

            if result.returncode != 0:
                logger.error(
                    f"[Video] FFmpeg failed: {result.stderr.decode('utf-8', errors='replace')[:200]}"
                )
                return [], meta

            # Parse raw RGB frames from stdout
            raw = result.stdout
            w, h = self.target_size
            frame_size = w * h * 3  # RGB24

            if len(raw) < frame_size:
                logger.warning("[Video] FFmpeg output too small for even one frame")
                return [], meta

            frames: List[np.ndarray] = []
            for i in range(0, len(raw), frame_size):
                if i + frame_size > len(raw):
                    break
                if len(frames) >= self.max_frames:
                    break

                frame_rgb = np.frombuffer(
                    raw[i:i + frame_size], dtype=np.uint8
                ).reshape(h, w, 3)

                # Convert RGB → BGR (OpenCV native format)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)

            logger.debug(f"[Video] FFmpeg: extracted {len(frames)} frames")
            return frames, meta

        except subprocess.TimeoutExpired:
            logger.error(
                f"[Video] FFmpeg timeout ({settings.VIDEO_FFMPEG_TIMEOUT}s) for: {video_path}"
            )
            return [], meta
        except FileNotFoundError:
            logger.error(
                "[Video] FFmpeg binary not found. Install with: brew install ffmpeg"
            )
            return [], meta
        except Exception as exc:
            logger.error(f"[Video] FFmpeg extraction failed: {exc}")
            return [], meta

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_metadata(self, cap: cv2.VideoCapture) -> VideoMetadata:
        """Extract metadata from an open VideoCapture object."""
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            fps=fps, duration=duration, frame_count=frame_count,
            width=width, height=height, codec=codec.strip(),
        )

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Resize frame to target dimensions for consistent keypoint detection.

        Why?
            - Standardised size ensures SIFT finds comparable keypoint densities
            - Smaller frames = faster SIFT + FLANN computation
            - Removes resolution as a confounding variable
        """
        if frame is None or frame.size == 0:
            return None
        try:
            resized = cv2.resize(
                frame, self.target_size, interpolation=cv2.INTER_AREA
            )
            return resized
        except Exception as exc:
            logger.error(f"[Video] Frame preprocessing failed: {exc}")
            return None


# ── Video Matcher ─────────────────────────────────────────────────────────────

class VideoMatcher:
    """
    Compares two videos using window-based best-frame matching.

    Strategy (NOT full O(n²) pairwise):
        For each frame_a[i]:
            Search only frame_b[i ± temporal_window] (default ±3)
            Find the best matching frame using SIFT (0.7) + Histogram (0.3)
            If best_score > VIDEO_FRAME_MATCH_THRESHOLD → count as matched
            Apply temporal consistency bonus if match is temporally aligned

        Final: video_similarity = matched_frames / total_frames_a

    Complexity: O(n × 2w) where w = temporal_window, NOT O(n × m)
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular imports
        from core.sift_matcher import SIFTMatcher
        from core.histogram import HistogramMatcher

        self._sift = SIFTMatcher()
        self._hist = HistogramMatcher()
        self._extractor = VideoFrameExtractor()

        logger.debug(
            f"[Video] Matcher init | window={settings.VIDEO_TEMPORAL_WINDOW} "
            f"weights=SIFT:{settings.VIDEO_SIFT_WEIGHT}/Hist:{settings.VIDEO_HIST_WEIGHT} "
            f"match_threshold={settings.VIDEO_FRAME_MATCH_THRESHOLD}"
        )

    def compare(
        self,
        video_a: Union[str, Path],
        video_b: Union[str, Path],
    ) -> VideoMatchResult:
        """
        Full video comparison pipeline.

        Args:
            video_a: Path to the first video.
            video_b: Path to the second video.

        Returns:
            VideoMatchResult with similarity score, per-frame scores, metadata.
        """
        start_time = time.time()

        # ── Step 1: Extract frames ────────────────────────────────────────────
        frames_a, meta_a = self._extractor.extract_frames(str(video_a))
        frames_b, meta_b = self._extractor.extract_frames(str(video_b))

        # Division by zero guard
        if not frames_a:
            return VideoMatchResult(
                metadata_a=meta_a, metadata_b=meta_b,
                processing_time=time.time() - start_time,
                error="No frames extracted from video A",
            )
        if not frames_b:
            return VideoMatchResult(
                total_frames_a=len(frames_a),
                metadata_a=meta_a, metadata_b=meta_b,
                processing_time=time.time() - start_time,
                error="No frames extracted from video B",
            )

        logger.info(
            f"[Video] Comparing {len(frames_a)} frames (A) "
            f"vs {len(frames_b)} frames (B)"
        )

        # ── Step 2: Window-based best-frame matching ──────────────────────────
        per_frame_scores: List[float] = []
        matched = 0
        window = settings.VIDEO_TEMPORAL_WINDOW

        for i, frame_a in enumerate(frames_a):
            best_score = 0.0
            best_j = -1

            # Temporal window: only compare nearby frames
            # This reduces complexity from O(n×m) to O(n×2w)
            j_start = max(0, i - window)
            j_end = min(len(frames_b), i + window + 1)

            for j in range(j_start, j_end):
                score = self._frame_similarity(frame_a, frames_b[j])

                if score > best_score:
                    best_score = score
                    best_j = j

                # Early exit: already found a very strong match
                if best_score >= settings.VIDEO_EARLY_EXIT_SCORE:
                    break

            # ── Temporal consistency bonus ─────────────────────────────────────
            # Reward when the best matching frame is temporally aligned
            # (within 2 positions of the source frame index)
            # This rewards same-timeline content and penalises random matches
            if best_j >= 0 and abs(i - best_j) < 2:
                best_score = min(1.0, best_score + settings.VIDEO_TEMPORAL_BONUS)
                logger.debug(
                    f"[Video] Frame {i}: temporal bonus applied "
                    f"(best_j={best_j}, delta={abs(i - best_j)})"
                )

            per_frame_scores.append(best_score)

            if best_score >= settings.VIDEO_FRAME_MATCH_THRESHOLD:
                matched += 1

            logger.debug(
                f"[Video] Frame {i}/{len(frames_a)}: "
                f"best_score={best_score:.4f} (matched_j={best_j})"
            )

        # ── Step 3: Final similarity ──────────────────────────────────────────
        similarity = matched / len(frames_a)

        elapsed = time.time() - start_time

        result = VideoMatchResult(
            similarity=similarity,
            matched_frames=matched,
            total_frames_a=len(frames_a),
            total_frames_b=len(frames_b),
            per_frame_scores=per_frame_scores,
            metadata_a=meta_a,
            metadata_b=meta_b,
            processing_time=elapsed,
        )

        logger.info(f"[Video] {result.summary()}")
        return result

    def _frame_similarity(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> float:
        """
        Compute composite similarity between two preprocessed frames.

        Weighted combination:
            frame_score = 0.7 × SIFT_score + 0.3 × Histogram_score

        SIFT-conditional histogram dampening is applied via the histogram
        module's built-in sift_score parameter.

        Why no pHash or OCR?
            - pHash: unreliable on individual video frames (compression artifacts)
            - OCR: too slow per-frame and rarely useful for video content

        Args:
            frame_a: Preprocessed BGR frame from video A.
            frame_b: Preprocessed BGR frame from video B.

        Returns:
            Composite score in [0.0, 1.0].
        """
        try:
            # SIFT structural matching
            sift_result = self._sift.match_arrays(frame_a, frame_b)
            sift_score = sift_result.score

            # Histogram colour matching (with SIFT-conditional dampening)
            hist_score = self._hist.compare(
                frame_a, frame_b, sift_score=sift_score
            )

            # Weighted composite: structure > colour
            composite = (
                settings.VIDEO_SIFT_WEIGHT * sift_score
                + settings.VIDEO_HIST_WEIGHT * hist_score
            )

            return max(0.0, min(1.0, composite))

        except Exception as exc:
            logger.error(f"[Video] Frame similarity failed: {exc}")
            return 0.0


# ── Module-level singletons ──────────────────────────────────────────────────
#   from core.video import video_matcher, frame_extractor
#   result = video_matcher.compare("a.mp4", "b.mp4")

frame_extractor = VideoFrameExtractor()
video_matcher = VideoMatcher()
