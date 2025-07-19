"""Interactive UI for the `diagnose_customer_issue` helper.

This lightweight Gradio interface lets users upload:
1. A PDF outlining the customer's problem.
2. One or more raw diagnostic data files (any text / log format).

It then invokes ``finder.diagnose_customer_issue`` and displays the LLM's
analysis.  The UI is intentionally minimal and can be expanded later.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import streamlit as st

from finder import diagnose_customer_issue


# ---------------------------------------------------------------------------
# Backend callback
# ---------------------------------------------------------------------------


def _run_diagnosis(
    # (kept for backward-compat if called programmatically)
    pdf_file: Path | None,
    raw_files: list[Path] | None,
):
    """Run diagnosis and yield interim status for a better UX."""
    if pdf_file is None or not raw_files:
        yield "Please upload both the PDF and raw data files."
        return

    # Interim message so the user sees an immediate response (acts as a loader).
    yield "⏳ Running diagnosis – this may take a minute..."

    pdf_path = Path(pdf_file.name)
    # Handle both temp file objects returned by Gradio (which have a `.name`)
    # and plain string/Path file representations.
    raw_paths = [Path(p.name) if hasattr(p, "name") else Path(p) for p in raw_files]

    try:
        # Provider is chosen via environment variable LLM_PROVIDER (or defaults)
        result = diagnose_customer_issue(pdf_path, raw_paths)
        yield result
    except Exception as exc:
        yield f"Error running diagnosis: {exc}"


# ---------------------------------------------------------------------------
# Streamlit UI launcher (supports multi-file upload out of the box)
# ---------------------------------------------------------------------------


def launch(host: str | None = None, port: int | None = None):  # noqa: D401 – keep signature stable
    """Launch a minimal Streamlit app for interactive diagnosis.

    Streamlit's ``st.file_uploader`` natively supports the ``accept_multiple_files``
    flag, which avoids the single-file limitation we encountered with the previous
    Gradio-based UI.  Running this script with ``streamlit run finder_ui.py`` will
    start the web interface.  The *host* and *port* parameters are kept for
    backward-compatibility but are unused because Streamlit manages those via CLI
    flags (``--server.port`` etc.).
    """

    st.set_page_config(page_title="Customer Issue Diagnosis")

    st.title("Customer Issue Diagnosis")

    pdf_file = st.file_uploader(
        "Problem description (PDF)",
        type=["pdf"],
        accept_multiple_files=False,
    )

    raw_files = st.file_uploader(
        "Raw diagnostic data files (txt/log/other)",
        type=["txt", "log", "csv", "json", "xml", "zip", "gz", "*"],
        accept_multiple_files=True,
    )

    if st.button("Diagnose"):
        if pdf_file is None or not raw_files:
            st.warning("Please upload both the PDF and at least one raw data file.")
            st.stop()

        with st.spinner("⏳ Running diagnosis – this may take a minute..."):
            tmp_dir = Path(tempfile.mkdtemp(prefix="finder_uploads_"))

            # Save uploaded PDF
            pdf_path = tmp_dir / pdf_file.name
            pdf_path.write_bytes(pdf_file.getbuffer())

            # Save raw files
            raw_paths: list[Path] = []
            for rf in raw_files:
                raw_path = tmp_dir / rf.name
                raw_path.write_bytes(rf.getbuffer())
                raw_paths.append(raw_path)

            try:
                result = diagnose_customer_issue(pdf_path, raw_paths)
                st.markdown(result)
            except Exception as exc:
                st.error(f"Error running diagnosis: {exc}")


if __name__ == "__main__":
    launch()
