"""Interactive UI for the `diagnose_customer_issue` helper.

This lightweight Gradio interface lets users upload:
1. A PDF outlining the customer's problem.
2. A raw diagnostic data file (any text / log format).

It then invokes ``finder.diagnose_customer_issue`` and displays the LLM's
analysis.  The UI is intentionally minimal and can be expanded later.
"""

from __future__ import annotations

import gradio as gr
from pathlib import Path

from finder import diagnose_customer_issue


# ---------------------------------------------------------------------------
# Backend callback
# ---------------------------------------------------------------------------


def _run_diagnosis(
    pdf_file: gr.File | None,
    raw_files: list[gr.File] | None,
):
    """Run diagnosis and yield interim status for a better UX."""
    if pdf_file is None or not raw_files:
        yield "Please upload both the PDF and raw data files."
        return

    # Interim message so the user sees an immediate response (acts as a loader).
    yield "⏳ Running diagnosis – this may take a minute..."

    pdf_path = Path(pdf_file.name)
    raw_paths = [Path(f.name) for f in raw_files]

    try:
        # Provider is chosen via environment variable LLM_PROVIDER (or defaults)
        result = diagnose_customer_issue(pdf_path, raw_paths)
        yield result
    except Exception as exc:
        yield f"Error running diagnosis: {exc}"


# ---------------------------------------------------------------------------
# UI launcher
# ---------------------------------------------------------------------------


def launch(host: str = "127.0.0.1", port: int = 7861):
    """Spin up the Gradio web UI."""

    with gr.Blocks(title="Customer Issue Diagnosis") as demo:
        gr.Markdown("# Customer Issue Diagnosis")

        with gr.Row():
            pdf_input = gr.File(label="Problem description (PDF)", file_types=[".pdf"])
            raw_input = gr.Files(label="Raw diagnostic data files (txt/log/other)")

        run_btn = gr.Button("Diagnose")
        output_md = gr.Markdown()

        run_btn.click(_run_diagnosis, inputs=[pdf_input, raw_input], outputs=output_md)

    demo.queue().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    launch()
