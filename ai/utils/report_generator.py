"""
utils/report_generator.py — Groq-powered AI Explanation
==========================================================

Generates a human-readable, professional explanation of the AI match
results using the Groq API (llama3-8b-8192).

Input: Final score and signal breakdown.
Output: Structured JSON containing the explanation and original scores.
"""

import os
import json
from loguru import logger
from groq import Groq

# Fetch API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client only if key exists
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def generate_report(match_data: dict) -> dict:
    """
    Calls Groq API to generate a professional explanation of the match result.
    If no API key is present, returns a fallback response.
    """
    if not GROQ_API_KEY or not client:
        return {
            "summary": "Match computed successfully.",
            "reasoning": "AI explanation unavailable (missing API key)."
        }

    try:
        score = match_data.get("score", 0.0)
        media_type = match_data.get("type", "image")
        breakdown = match_data.get("breakdown", {})

        # Construct the prompt
        prompt = (
            f"You are an AI Media Matching System analyzer. "
            f"I will provide you with the matching scores between a query {media_type} and a reference file.\n"
            f"- Overall Similarity Score: {score:.2f} (0.0 to 1.0)\n"
        )

        if media_type == "image":
            prompt += (
                f"- pHash (Global Structure): {breakdown.get('phash', 0.0):.2f}\n"
                f"- SIFT (Local Keypoints/Structure): {breakdown.get('sift', 0.0):.2f}\n"
                f"- Histogram (Color Palette): {breakdown.get('histogram', 0.0):.2f}\n"
                f"- OCR (Text Jaccard): {breakdown.get('ocr', 0.0):.2f}\n\n"
                f"Note: If the overall score is very high (e.g. 1.0) but SIFT/Histogram/OCR are 0.0, it means the system "
                f"detected an exact duplicate via pHash and safely skipped the heavier computations (Early Exit).\n"
            )
            prompt += "\nExplain in 2-3 short, professional sentences whether this is likely a match, a modified copy, or completely different, and briefly justify it based on the signals above. Do not hallucinate numbers."
        else:
            prompt += (
                "For videos, we use temporal frame-by-frame SIFT and Histogram matching.\n"
                "Explain in 2-3 short, professional sentences whether this is likely a matching video or different, based on the overall similarity score."
            )

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional forensic media analyst. Provide only the concise explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=150,
        )

        explanation = chat_completion.choices[0].message.content.strip()

        # Build the structured response
        return {
            "score": score,
            "type": media_type,
            "breakdown": breakdown,
            "explanation": explanation
        }

    except Exception as e:
        logger.error(f"[ReportGenerator] Failed to generate Groq report: {e}")
        # Fallback response so the API doesn't fail
        return {
            "score": match_data.get("score", 0.0),
            "type": match_data.get("type", "unknown"),
            "breakdown": match_data.get("breakdown", {}),
            "explanation": "AI report generation failed due to an API error."
        }
