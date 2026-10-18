"""Gemini LLM factory for Job Autopilot (replaces Groq)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# Allow importing shared helper when running from this folder
_SHARED = Path(__file__).resolve().parents[2] / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from gemini_chat import get_chat_llm  # noqa: E402


def as_text(content) -> str:
    """Gemini sometimes returns content as a list of blocks instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                text = getattr(block, "text", None)
                parts.append(text if isinstance(text, str) else str(block))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def get_llm(temperature: float = 0.3):
    """Return shared Gemini chat model; ignore Streamlit secrets for smoke tests."""
    # Prefer env; Streamlit secrets are optional
    try:
        import streamlit as st

        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            os.environ.setdefault("GEMINI_API_KEY", key)
    except Exception:
        pass
    return get_chat_llm(temperature=temperature)
