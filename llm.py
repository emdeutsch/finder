"""Helper functions to instantiate an LLM based on environment variables.

Moved from ``veo_agent.agent.content_agent`` so the old ``agent`` package can be removed.
"""

from __future__ import annotations

import os

# Load environment variables from a .env file if present so that users can
# configure credentials without exporting them in the shell.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:  # pragma: no cover – fallback if python-dotenv isn't installed
    pass

from typing import Literal

from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Private model helpers
# ---------------------------------------------------------------------------


def _vertex_llm() -> ChatVertexAI:
    """Return a Vertex AI chat model (Bison / Gemini-2.5)."""
    return ChatVertexAI(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
        model_name=os.getenv("GEMINI_MODEL", "chat-bison@001"),
        temperature=0.3,
        max_output_tokens=1024,
    )


def _gemini_api_llm() -> ChatGoogleGenerativeAI:
    """Return a Gemini model instantiated via the public Google AI Studio API key.

    NOTE: The Gemini API currently returns **empty content** (finish_reason = MAX_TOKENS
    with output_tokens = 0) if ``max_output_tokens`` is supplied and the model hits
    that limit – even if the limit is well below the model's actual maximum.
    To avoid this bug we **omit** the parameter entirely, letting the backend decide
    an appropriate value.
    """
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2p5-flash"),
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3,
    )


def _openai_llm() -> ChatOpenAI:
    """Return an OpenAI GPT-4o (or 4o-mini) chat model instance."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.3,
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_llm(
    provider: Literal["vertex", "gemini_api", "openai"] | None = None,
):
    """Return a LangChain chat model based on runtime configuration.

    The selection order is:
    1. Explicit *provider* argument if given.
    2. Environment variable ``LLM_PROVIDER``.
    3. Default to Vertex AI.

    Parameters
    ----------
    provider : Literal["vertex", "gemini_api", "openai"] | None
        Provider override. If ``None`` (default) follow env vars / defaults.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "vertex").lower()  # type: ignore

    # Emit a short debug line so users can verify which backend is selected.
    print(f"[LLM] Using provider: {provider}")

    if provider == "gemini_api":
        return _gemini_api_llm()
    if provider == "openai":
        return _openai_llm()
    # fallback → Vertex AI
    return _vertex_llm()


__all__ = ["get_llm"]
