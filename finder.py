"""Utility for diagnosing customer issues by combining PDF descriptions with raw diagnostic data.

This helper provides an LLM-powered analysis routine that:
1. Converts the provided PDF file to plain text.
2. Reads the raw diagnostic data file as plain text.
3. Feeds both pieces of context into an LLM (via veo_agent.agents.llm.get_llm) with a
   carefully crafted prompt asking for a root-cause analysis.

The function defined here is *not* invoked automatically; integration will be handled
in a subsequent step.
"""

from __future__ import annotations

import pathlib
from typing import Literal, Sequence

from llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

# Optional dependencies for OCR of images inside PDFs
try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover
    convert_from_path = None  # type: ignore[assignment]

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore

# PDF text extraction dependency
try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_ocr_text(pdf_path: str | pathlib.Path) -> str:
    """Return text extracted via OCR from images in the PDF.

    This requires both *pdf2image* (to render PDF pages as images) and
    *pytesseract* (to perform OCR).  If either dependency is missing or an
    error occurs, an empty string is returned.
    """
    if convert_from_path is None or pytesseract is None:
        return ""

    try:
        images = convert_from_path(str(pdf_path), fmt="png")
    except Exception:
        return ""

    ocr_chunks: list[str] = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img)
        except Exception:
            text = ""
        if text:
            ocr_chunks.append(text.strip())

    return "\n\n".join(ocr_chunks)


def _extract_pdf_text(pdf_path: str | pathlib.Path) -> str:
    """Return all textual contents extracted from *pdf_path*.

    If PyPDF2 is unavailable or the file cannot be parsed, an empty string is
    returned so the analysis can proceed (the raw data may still be useful).
    """
    if PdfReader is None:
        text_content = ""
    else:
        try:
            reader = PdfReader(str(pdf_path))
            text_content = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception:
            text_content = ""

    # Attempt OCR on embedded images/pages for additional context
    ocr_text = _extract_ocr_text(pdf_path)

    if ocr_text:
        return f"{text_content}\n\n{ocr_text}" if text_content else ocr_text
    return text_content


def _read_raw_data(raw_path: str | pathlib.Path) -> str:
    """Return the raw data as a UTF-8 string (falling back to Latin-1 on decode errors)."""
    path = pathlib.Path(raw_path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        # Last-ditch effort: read as bytes then decode.
        try:
            return path.read_bytes().decode("latin-1", errors="replace")
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose_customer_issue(
    pdf_path: str | pathlib.Path,
    raw_data_path: str | pathlib.Path | Sequence[str | pathlib.Path],
    *,
    provider: Literal["vertex", "gemini_api", "openai"] | None = None,
) -> str:
    """Analyse the customer's issue based on the PDF description and raw diagnostic data.

    Parameters
    ----------
    pdf_path : str | pathlib.Path
        Path to the PDF that summarises the customer's problem and potential causes.
    raw_data_path : str | pathlib.Path
        Path to a file containing low-level diagnostic information (format agnostic).
    provider : str, optional
        Force a specific LLM backend.  If omitted, falls back to environment defaults.

    Returns
    -------
    str
        A thorough, human-readable explanation of *why* the customer is experiencing
        the issue, including references to both the PDF content and the raw data.
    """
    pdf_text = _extract_pdf_text(pdf_path)
    # Support a single path or a list/tuple of paths for raw diagnostic files.
    if isinstance(raw_data_path, (list, tuple)):
        raw_parts: list[str] = []
        for idx, path in enumerate(raw_data_path, start=1):
            text = _read_raw_data(path)
            raw_parts.append(
                f"===== RAW FILE {idx} ({path}) START =====\n{text}\n===== RAW FILE {idx} END ====="
            )
        raw_text = "\n\n".join(raw_parts)
    else:
        raw_text = _read_raw_data(raw_data_path)

    # Build LLM messages
    system_prompt = (
        "You are a senior support engineer. A customer has provided a PDF describing "
        "their problem, as well as various hypotheses about what might be causing it. "
        "Your task is to determine the root cause of the issue and explain it in clear, lay-person terms."
    )

    user_prompt = (
        "The PDF below outlines the customer's problem and lists several possible causes.\n\n"
        "----- PDF START -----\n"
        f"{pdf_text}\n"
        "----- PDF END -----\n\n"
        "Below is additional raw data that should allow you to pinpoint the precise cause.\n\n"
        "----- RAW DATA START -----\n"
        f"{raw_text}\n"
        "----- RAW DATA END -----\n\n"
        "Using ONLY the information above, please diagnose the customer's issue and thoroughly "
        "explain why it is occurring. Provide a concise summary followed by a detailed technical analysis. "
        "If the information is insufficient, state what additional data would be required."
    )

    # ------------------------------------------------------------------
    # Invoke LLM
    # ------------------------------------------------------------------

    llm = get_llm(provider=provider)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    # Print full prompt for debugging purposes
    print("\n===== LLM PROMPT (SYSTEM) =====\n", system_prompt, sep="")
    print("\n===== LLM PROMPT (USER) =====\n", user_prompt, sep="")

    response = llm.invoke(messages)  # type: ignore[arg-type]

    # Extract text from response and log it
    if isinstance(response, str):
        content = response
    else:
        content = str(getattr(response, "content", "")) or str(response)

    # Print the raw response object for additional debugging
    print("\n===== LLM RAW OBJECT =====\n", repr(response), sep="")
    print("\n===== LLM RESPONSE =====\n", content, sep="")

    return content
