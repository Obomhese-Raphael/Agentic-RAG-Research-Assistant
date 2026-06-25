import asyncio
from pathlib import Path
import time
import os

import streamlit as st
import inngest
from dotenv import load_dotenv
import requests

load_dotenv()

st.set_page_config(page_title="Agentic RAG Research Assistant",
                   page_icon="🧠", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="agentic-rag-app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    ids = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    return ids[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                output = run.get("output")
                if output:
                    return output

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# UI: Ingestion
# ---------------------------------------------------------------------------

st.title("🧠 Agentic RAG Research Assistant")
st.caption("Upload PDFs, then ask questions. The system decides for itself whether to search your documents.")

st.divider()
st.subheader("1. Upload a PDF")

uploaded = st.file_uploader(
    "Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        asyncio.run(send_rag_ingest_event(path))
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption(
        "Give it a few seconds to finish chunking and embedding before asking questions about it.")

st.divider()
st.subheader("2. Ask a question")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve",
                            min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Routing, retrieving, and generating an answer..."):
            event_id = asyncio.run(send_rag_query_event(
                question.strip(), int(top_k)))
            output = wait_for_run_output(event_id)

        answer = output.get("answer", "")
        used_retrieval = output.get("used_retrieval", False)
        confidence = output.get("confidence")
        sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")

        # ---- Agentic internals panel ----
        st.divider()
        st.markdown("##### How this answer was produced")

        col1, col2 = st.columns(2)
        with col1:
            if used_retrieval:
                st.info(
                    "🔍 **Retrieval used** — this question was routed to search your documents.")
            else:
                st.info(
                    "💬 **No retrieval needed** — answered directly without searching documents.")

        with col2:
            if confidence == "high":
                st.success(
                    "✅ **High confidence** — retrieved content closely matched the question.")
            elif confidence == "low":
                st.warning(
                    "⚠️ **Low confidence** — retrieved content was a weak match. Treat this answer with caution.")

        if sources:
            st.markdown("**Sources cited:**")
            for s in sources:
                st.write(f"- {s}")
