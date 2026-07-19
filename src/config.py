"""
Centralized configuration.

Loads secrets from a local .env file (never hardcode API keys in source).
Import constants from here rather than reading os.environ in multiple places,
so there is exactly one place to change if a setting needs to move (e.g. to
a different LLM provider or a different model name).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("[WARNING]: OPENAI_API_KEY is missing. Check your .env file setup!")

# --- LLM settings ---------------------------------------------------------
# gpt-4o-mini is a good balance of cost and quality for structured JSON output.
# Swap to "gpt-4o" for higher quality, or replace this whole config with a
# Gemini client if you'd rather use Google's API.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- App-level defaults ---------------------------------------------------
SUPPORTED_SPORTS = ["Cricket", "Football", "Badminton", "Tennis", "Basketball"]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
NUM_QUESTIONS = 4          # how many quiz questions to generate per request
CHROMA_RESULTS_PER_QUERY = 3
WEB_SEARCH_RESULTS = 3

# Below this token-overlap ratio between an explanation and the retrieved
# context, a question gets flagged as "unverified" in the grounding check.
GROUNDING_OVERLAP_THRESHOLD = 0.25
