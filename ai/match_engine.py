"""
match_engine.py — The Final Brain of the System
=================================================

Orchestrates the entire media matching pipeline by routing requests,
combining signals (pHash, SIFT, Histogram, OCR), and applying adaptive
reliability weighting.

Usage:
    from match_engine import MediaMatcher
    matcher = MediaMatcher()
    results = matcher.match_image("query.jpg")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from config import settings
from core.fingerprint import fingerprinter
from core.histogram import histogram_matcher
from core.ocr import ocr_matcher
from core.sift_matcher import sift_matcher
from core.reliability import assess_reliability, apply_reliability_weights
from core.video import video_matcher, VideoMatchResult


@dataclass
class MatchBreakdown:
    """Detailed breakdown of individual signal scores and weights."""
    phash_score: float = 0.0
    sift_score: float = 0.0
    hist_score: float = 0.0
    ocr_score: float = 0.0
    
    phash_weight: float = 0.0
    sift_weight: float = 0.0
    hist_weight: float = 0.0
    ocr_weight: float = 0.0

    def summary(self) -> str:
        return (
            f"Breakdown: "
            f"pHash({self.phash_score:.2f} @ {self.phash_weight:.2f}) | "
            f"SIFT({self.sift_score:.2f} @ {self.sift_weight:.2f}) | "
            f"Hist({self.hist_score:.2f} @ {self.hist_weight:.2f}) | "
            f"OCR({self.ocr_score:.2f} @ {self.ocr_weight:.2f})"
        )


@dataclass
class MatchResult:
    """Final output for a single comparison."""
    corpus_item: str
    composite_score: float
    is_match: bool
    breakdown: MatchBreakdown
    processing_time: float
    error: Optional[str] = None


class MediaMatcher:
    """
    Central orchestration layer.
    Combines all signals with adaptive reliability weighting.
    """

    def __init__(self, corpus_dir: Union[str, Path] = settings.CORPUS_DIR):
        self.corpus_dir = Path(corpus_dir)
        
        # Base weights for images
        self.base_weights = {
            "phash": settings.WEIGHT_PHASH,
            "sift": settings.WEIGHT_SIFT,
            "hist": settings.WEIGHT_HISTOGRAM,
            "ocr": settings.WEIGHT_OCR,
        }

        logger.info(f"[Matcher] Initialised | corpus={self.corpus_dir}")

    # ── Routing ───────────────────────────────────────────────────────────────

    def match(self, query_path: Union[str, Path]) -> List[Union[MatchResult, VideoMatchResult]]:
        """
        Auto-routes to match_image or match_video based on extension.
        Matches against the entire corpus.
        """
        path = Path(query_path)
        if not path.exists():
            logger.error(f"[Matcher] Query file not found: {path}")
            return []

        if settings.is_supported_image(path.name):
            return self.match_image_corpus(path)
        elif settings.is_supported_video(path.name):
            return self.match_video_corpus(path)
        else:
            logger.error(f"[Matcher] Unsupported file type: {path}")
            return []

    # ── Image Pipeline ────────────────────────────────────────────────────────

    def compare_images(
        self, query_path: Union[str, Path], corpus_path: Union[str, Path]
    ) -> MatchResult:
        """
        Compare two images and return a unified composite score.
        """
        start_time = time.time()
        
        query = Path(query_path)
        corpus = Path(corpus_path)
        
        logger.debug(f"[Matcher] Comparing: {query.name} vs {corpus.name}")

        # ── 1. pHash (Fast Filter) ──
        hash_q = fingerprinter.compute_hash(query)
        hash_c = fingerprinter.compute_hash(corpus)

        if not hash_q or not hash_c:
            return MatchResult(
                corpus_item=corpus.name, composite_score=0.0, is_match=False,
                breakdown=MatchBreakdown(), processing_time=time.time() - start_time,
                error="pHash computation failed"
            )

        phash_score = fingerprinter.similarity_score(hash_q, hash_c)

        # Early exit check (if enabled and score is VERY high or VERY low)
        if settings.ENABLE_EARLY_EXIT:
            if phash_score >= settings.EARLY_EXIT_PHASH_SCORE:
                logger.info(f"[Matcher] Early exit STRONG match: {corpus.name}")
                breakdown = MatchBreakdown(phash_score=phash_score, phash_weight=1.0)
                return MatchResult(
                    corpus_item=corpus.name, composite_score=phash_score,
                    is_match=True, breakdown=breakdown,
                    processing_time=time.time() - start_time
                )
            
            gate = fingerprinter.soft_gate(hash_q, hash_c)
            if gate == "SKIP":
                logger.debug(f"[Matcher] Early exit SKIP: {corpus.name}")
                breakdown = MatchBreakdown(phash_score=phash_score, phash_weight=1.0)
                return MatchResult(
                    corpus_item=corpus.name, composite_score=phash_score,
                    is_match=False, breakdown=breakdown,
                    processing_time=time.time() - start_time
                )

        # ── 2. Heavy Signals ──
        # SIFT
        sift_res = sift_matcher.match(query, corpus)
        sift_score = sift_res.confidence_score
        
        # Histogram (with SIFT dampening)
        hist_score = histogram_matcher.compare(query, corpus, sift_score=sift_score)
        
        # OCR
        tokens_q = ocr_matcher.extract_tokens(query) or set()
        tokens_c = ocr_matcher.extract_tokens(corpus) or set()
        ocr_score = ocr_matcher.jaccard_similarity(tokens_q, tokens_c)

        # ── 3. Adaptive Reliability Weighting ──
        # Load one image to get area for histogram reliability
        import cv2
        img = cv2.imread(str(query), cv2.IMREAD_GRAYSCALE)
        area = img.shape[0] * img.shape[1] if img is not None else 0
        
        rel = assess_reliability(
            ocr_token_count_a=len(tokens_q),
            ocr_token_count_b=len(tokens_c),
            image_area=area,
            sift_keypoints=sift_res.total_keypoints,
            is_video_frame=False
        )
        
        weights = apply_reliability_weights(self.base_weights, rel)

        # ── 4. Final Composite Score ──
        final_score = (
            weights.get("phash", 0) * phash_score +
            weights.get("sift", 0) * sift_score +
            weights.get("hist", 0) * hist_score +
            weights.get("ocr", 0) * ocr_score
        )

        is_match = final_score >= settings.MATCH_CONFIDENCE_THRESHOLD

        breakdown = MatchBreakdown(
            phash_score=phash_score, sift_score=sift_score,
            hist_score=hist_score, ocr_score=ocr_score,
            phash_weight=weights.get("phash", 0),
            sift_weight=weights.get("sift", 0),
            hist_weight=weights.get("hist", 0),
            ocr_weight=weights.get("ocr", 0)
        )

        result = MatchResult(
            corpus_item=corpus.name,
            composite_score=final_score,
            is_match=is_match,
            breakdown=breakdown,
            processing_time=time.time() - start_time
        )
        
        logger.info(f"[Matcher] Image Match: {corpus.name} | Score={final_score:.4f} | {breakdown.summary()}")
        return result

    def match_image_corpus(self, query_path: Union[str, Path]) -> List[MatchResult]:
        """Match an image against all supported images in the corpus."""
        results = []
        if not self.corpus_dir.exists():
            return results
            
        for corpus_file in self.corpus_dir.iterdir():
            if corpus_file.is_file() and settings.is_supported_image(corpus_file.name):
                res = self.compare_images(query_path, corpus_file)
                results.append(res)
                
                # Global early exit if we found a very strong match
                if res.composite_score >= settings.EARLY_EXIT_MIN_COMPOSITE:
                    logger.info(f"[Matcher] Global early exit triggered by {corpus_file.name}")
                    break
                    
        results.sort(key=lambda x: x.composite_score, reverse=True)
        return results[:settings.TOP_K_RESULTS]

    # ── Video Pipeline ────────────────────────────────────────────────────────

    def match_video_corpus(self, query_path: Union[str, Path]) -> List[VideoMatchResult]:
        """Match a video against all supported videos in the corpus."""
        results = []
        if not self.corpus_dir.exists():
            return results
            
        # Add filename to VideoMatchResult dynamically for context
        for corpus_file in self.corpus_dir.iterdir():
            if corpus_file.is_file() and settings.is_supported_video(corpus_file.name):
                res = video_matcher.compare(query_path, corpus_file)
                # Inject corpus name into error field temporarily or create a wrapper
                # For now, just logging it
                logger.info(f"[Matcher] Video Match vs {corpus_file.name}: Sim={res.similarity:.4f}")
                results.append((corpus_file.name, res))
                
        # Sort by similarity
        results.sort(key=lambda x: x[1].similarity, reverse=True)
        
        # We can return a dict or wrap it
        return results[:settings.TOP_K_RESULTS]
