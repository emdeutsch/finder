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
from typing import Literal, Sequence, cast, Any, TYPE_CHECKING

import base64
from io import BytesIO

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


def _extract_pdf_page_images(pdf_path: str | pathlib.Path, *, max_pages: int | None = None) -> list[str]:
    """Render each PDF page to a PNG data-URL string for Gemini vision models.

    Parameters
    ----------
    pdf_path : str | pathlib.Path
        Path to the PDF file.
    max_pages : int | None, optional
        Optional limit on the number of pages converted (to respect API payload limits).

    Returns
    -------
    list[str]
        A list of ``data:image/png;base64,...`` strings – one for each rendered page.
    """
    if convert_from_path is None:
        # pdf2image missing → cannot extract page images.
        return []

    try:
        images = convert_from_path(str(pdf_path), fmt="png", dpi=200)
    except Exception:
        return []

    if max_pages is not None:
        images = images[: max_pages]

    data_urls: list[str] = []
    for img in images:
        buffer = BytesIO()
        try:
            img.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            data_urls.append(f"data:image/png;base64,{encoded}")
        except Exception:
            # Skip problematic pages but continue processing others.
            continue

    return data_urls


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
        raw_text = _read_raw_data(cast(str | pathlib.Path, raw_data_path))

    # ------------------------------------------------------------------
    # Build prompts – handle Gemini vision models specially so we can attach images
    # ------------------------------------------------------------------

    system_prompt = (
        "You are a senior support engineer. A customer has provided a PDF describing "
        "their problem, as well as various hypotheses about what might be causing it. "
        "Your task is to determine the root cause of the issue and explain it in clear, lay-person terms."
    )

    user_prompt = (
        "The full plain-text extraction of the PDF appears below, followed by the rendered images for "
        "each PDF page.  The images capture visual layout, graphics, logos, or any other information that "
        "may be missing from the raw text.  Use BOTH the text and the images together to understand the "
        "document in context.\n\n"
        "----- PDF TEXT START -----\n"
        f"{pdf_text}\n"
        "----- PDF TEXT END -----\n\n"
        "After the text you will find one image block per page, wrapped with the markers \"PDF PAGE N IMAGE START/END\".\n\n"
        "The customer also supplied raw diagnostic data which appears after the images.\n\n"
        "----- RAW DATA START -----\n"
        f"{raw_text}\n"
        "----- RAW DATA END -----\n\n"
        "Using ONLY the information provided (text + images + raw data), diagnose the customer's issue. "
        "Provide a concise summary, then a detailed technical analysis.  If the information is insufficient, "
        "state what additional data would help."
    )

    llm = get_llm(provider=provider)

    # Detect whether the selected LLM backend is Gemini *and* supports images.
    # We check either an explicit provider flag or the instantiated class name.
    if TYPE_CHECKING:
        # Only for static type checking – avoid runtime dependency if package missing.
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore  # pragma: no cover

    # Use string comparison to avoid hard import of ChatGoogleGenerativeAI for runtime.
    using_gemini_vision = False
    if (provider or "").lower() == "gemini_api":
        using_gemini_vision = True
    else:
        # Check class name to detect Gemini model without importing module.
        if llm.__class__.__name__ == "ChatGoogleGenerativeAI":
            using_gemini_vision = True

    # Prepare placeholder for messages to avoid redefinition issues with static type checkers.
    messages: list[Any]

    # Gemini vision models require a *single* HumanMessage and support images.
    if using_gemini_vision:
        # Combine system + user text, then append images with page context.
        combined_text = f"{system_prompt}\n\n{user_prompt}"

        content_parts: list[Any] = [{"type": "text", "text": combined_text}]

        # Attach page images – limit to 16 to respect Gemini payload limits.
        page_images = _extract_pdf_page_images(pdf_path, max_pages=16)
        for idx, data_url in enumerate(page_images, start=1):
            # Provide clear delimiters so the model knows the source page.
            content_parts.append({"type": "text", "text": f"----- PDF PAGE {idx} IMAGE START -----"})
            content_parts.append({"type": "image_url", "image_url": data_url})
            content_parts.append({"type": "text", "text": f"----- PDF PAGE {idx} IMAGE END -----"})

        # Cast to satisfy strict typing – runtime will accept the structure.
        messages = [HumanMessage(content=content_parts)]  # type: ignore[arg-type]

        # Debug logging – show page markers so users see where images go.
        print("\n===== LLM PROMPT (COMBINED TEXT + IMAGE MARKERS) =====\n", combined_text, sep="")
        print(f"\n[LLM] Attached {len(page_images)} page image(s). Markers are included in the prompt after the text block.")
    else:
        # Original two-message flow for providers that do not support image input.
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]  # type: ignore[arg-type]

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
