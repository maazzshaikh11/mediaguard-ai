from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from config import settings
from match_engine import MediaMatcher
from utils.report_generator import generate_report

router = APIRouter()
matcher = MediaMatcher()

# Global thread pool for non-blocking execution of CPU-heavy tasks
executor = ThreadPoolExecutor(max_workers=4)

def run_analysis(file_path: str) -> dict:
    """
    Synchronous analysis pipeline to be run in a thread pool.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("File not found")

    is_image = settings.is_supported_image(path.name)
    is_video = settings.is_supported_video(path.name)

    if not is_image and not is_video:
        raise ValueError("Unsupported media format")

    # Run matcher against the corpus
    results = matcher.match(path)

    if not results:
        # No matches found or extraction failed
        return {
            "score": 0.0,
            "type": "image" if is_image else "video",
            "breakdown": {"phash": 0.0, "sift": 0.0, "histogram": 0.0, "ocr": 0.0},
            "explanation": "No matching content found in the corpus."
        }

    # Grab the best match
    best_match = results[0]

    media_type = "image" if is_image else "video"
    
    # Format breakdown depending on result type
    if media_type == "image":
        breakdown = {
            "phash": best_match.breakdown.phash_score,
            "sift": best_match.breakdown.sift_score,
            "histogram": best_match.breakdown.hist_score,
            "ocr": best_match.breakdown.ocr_score
        }
        score = best_match.composite_score
    else:
        # VideoMatchResult
        breakdown = {
            "phash": 0.0,
            "sift": best_match[1].similarity, # Video uses composite in similarity
            "histogram": best_match[1].similarity,
            "ocr": 0.0
        }
        score = best_match[1].similarity

    match_data = {
        "score": score,
        "type": media_type,
        "breakdown": breakdown
    }

    # Generate Groq Report
    final_response = generate_report(match_data)
    
    return final_response


@router.post("/analyze")
async def analyze_media(file_path: str = Form(...)):
    """
    Analyze an existing file path asynchronously using a thread pool.
    """
    try:
        loop = asyncio.get_running_loop()
        # Run blocking code in thread pool
        result = await loop.run_in_executor(executor, run_analysis, file_path)
        return JSONResponse(content=result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found on server")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")


@router.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Upload a file, save it to uploads directory, and analyze it.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    is_image = settings.is_supported_image(file.filename)
    is_video = settings.is_supported_video(file.filename)
    
    if not is_image and not is_video:
        raise HTTPException(status_code=400, detail="Unsupported media format")

    # Generate unique filename to avoid collisions
    ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = settings.UPLOAD_DIR / unique_name

    try:
        # Save file asynchronously
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File uploaded successfully: {save_path}")

        # Dispatch to analysis pipeline
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_analysis, str(save_path))
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload/Analyze failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Optional: cleanup file after processing if not needed anymore
        # if save_path.exists():
        #     save_path.unlink()
        pass


@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "service": "MediaGuard AI Matching Engine"}
