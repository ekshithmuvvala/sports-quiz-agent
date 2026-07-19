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


def _get_openai_key():
    """
    Reads the API key from a local .env file (for running on your own
    machine) or from Streamlit's secrets manager (for Streamlit Community
    Cloud, which has no .env file -- secrets are set in the app dashboard
    instead). Whichever is present wins; local .env is checked first.
    """
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    try:
        import streamlit as st
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


# --- API keys -----------------------------------------------------------
OPENAI_API_KEY = _get_openai_key()

if not OPENAI_API_KEY:
    print(
        "[WARNING]: OPENAI_API_KEY is missing. Set it in a local .env file, "
        "or in Streamlit Cloud under Settings -> Secrets."
    )