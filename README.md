# Agentic RAG Research Assistant

A personal research tool that lets you point an LLM at a folder of PDFs and ask questions about them — but unlike basic RAG, it **decides whether retrieval is even necessary before searching**, **cites the exact document and page** behind every answer, and **flags when its own retrieval results don't actually support the question** instead of confidently guessing.

Built to go beyond a tutorial RAG clone by adding the three things most RAG demos skip: a reasoning step before retrieval, traceable citations, and a measured evaluation of how well the system actually performs — not just "it worked when I tried it."

---

## Why this exists

Most RAG tutorials retrieve unconditionally on every query and trust whatever comes back. That breaks down in two predictable ways: it searches documents for questions that don't need it (wasted latency, irrelevant context polluting the prompt), and it stays silent about *how good* a retrieval actually was — so a weak, barely-relevant match gets treated exactly the same as a strong one, and the model answers confidently either way.

This project adds two decision points around the standard retrieve-then-generate loop:

1. **Before retrieval:** a routing step asks "does this question actually require searching the documents, or can it be answered directly?" — aware of which documents are currently loaded, so it doesn't skip retrieval just because the model already has general knowledge of a famous text.
2. **After retrieval:** a relevance check asks "does what I just retrieved actually answer this question?" and adjusts the final answer's tone accordingly — hedging explicitly when the context doesn't hold up, rather than answering as if it's sure.

## Architecture

```
                    ┌─────────────────────┐
   User question →  │   ROUTING STEP       │  Groq call, aware of which
                    │   (router.py)        │  documents are currently loaded
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
           needs retrieval           no retrieval needed
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ EMBED + SEARCH    │      │ Answer directly   │
        │ (Qdrant)          │      │ from the LLM,      │
        └────────┬─────────┘      │ no documents       │
                  │                │ touched at all      │
                  ▼                └──────────────────┘
        ┌──────────────────┐
        │ RELEVANCE CHECK   │  LLM judges whether the retrieved
        │ (confidence_judge)│  chunks actually answer the question
        └────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
    relevant            not relevant
        │                   │
        ▼                   ▼
┌──────────────┐   ┌──────────────────────┐
│ Generate      │   │ Generate answer,      │
│ answer with   │   │ but explicitly hedge  │
│ citations     │   │ and explain what the  │
│ (doc + page)  │   │ document actually     │
└──────────────┘   │ covers instead         │
                    └──────────────────────┘
```

Every step above is a durable Inngest step — if the process crashes mid-pipeline, completed steps aren't re-run, only the failed one retries.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | [Inngest](https://www.inngest.com/) | Durable, retry-safe multi-step workflows — each pipeline stage (route, retrieve, judge, generate) is an independently retryable step |
| Vector store | [Qdrant](https://qdrant.tech/) | Local, fast, simple payload filtering for source/page metadata |
| Chunking | [LlamaIndex](https://www.llamaindex.ai/) | Sentence-aware splitting, page-level metadata out of the box via `PDFReader` |
| Embeddings | [HuggingFace `BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | Runs locally, no API cost, 384-dim |
| Inference | [Groq](https://groq.com/) (`openai/gpt-oss-20b`) | Fast inference for routing, relevance judging, and final generation |
| API | FastAPI | Serves the Inngest function handlers |
| UI | Streamlit | Upload + query interface, surfaces routing/confidence/sources to the user |

## What makes this more than a tutorial clone

- **Routing is context-aware, not just keyword-based.** The router is explicitly told which documents are currently loaded in Qdrant before deciding whether to search — so it doesn't skip retrieval on a question about a well-known text just because the model already half-knows the answer from training data. (See **Evaluation** below for a real example of this going wrong before the fix.)
- **Citations carry page numbers, not just filenames.** Every retrieved chunk is tagged with its source page during ingestion, and citations are deduplicated by document+page pair so multi-chunk answers don't repeat the same source five times.
- **Confidence is judged by content, not just vector distance.** An early version of this project used raw cosine similarity scores against a fixed threshold to flag low-confidence retrievals. The evaluation harness below caught that this didn't actually work — irrelevant and relevant matches overlapped in score range on this corpus. The system now uses a second, dedicated LLM call to judge whether retrieved content actually addresses the question, rather than relying on embedding distance alone.
- **There's a real evaluation harness**, not just manual spot-checking. `eval/run_eval.py` runs a labeled question set against the live pipeline and reports routing accuracy as a measured percentage.

## Evaluation

`eval/eval_set.json` contains 16 hand-written questions spanning two ingested document types (a historical text and a technical document set), deliberately including adversarial cases — questions phrased to *sound* document-related without actually being covered in the text.

Running `eval/run_eval.py` against the live pipeline produced:

- **Routing accuracy: 15/16 (93.75%)** — the one miss was an ambiguous conversational follow-up ("explain that in simpler terms") that the router can't resolve correctly without conversation memory, which is a known, reasonable limitation rather than a bug.
- **A genuine negative finding:** raw similarity scores from `bge-small-en-v1.5` did *not* cleanly separate relevant from irrelevant matches on this corpus — scores for clearly irrelevant questions (0.66–0.72) overlapped directly with scores for clearly relevant ones (0.66–0.83). No single fixed threshold could have worked. This is documented rather than hidden, and is what motivated replacing the threshold-based confidence check with an LLM-judged relevance check (see Architecture above).

This is the kind of result a real eval harness is supposed to produce — not every metric came back clean, and the failure itself is documented as a design decision rather than papered over.

## Setup

```bash
git clone <your-repo-url>
cd agentic-rag-research-assistant

uv venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

uv add fastapi inngest llama-index-core llama-index-embeddings-huggingface \
       llama-index-llms-groq llama-index-readers-file openai python-dotenv \
       qdrant-client streamlit uvicorn
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Start Qdrant (requires Docker):
```bash
docker run -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Start the Inngest dev server:
```bash
npx inngest-cli@latest dev
```

Start the FastAPI app (in a separate terminal):
```bash
python -m uvicorn main:app --reload --port 8000
```

Launch the UI:
```bash
streamlit run streamlit_app.py
```

Run the evaluation harness:
```bash
python eval/run_eval.py
```

## Known limitations

- Relevance judging adds an extra LLM call per retrieval-based query, increasing latency and cost slightly compared to a single-pass system.
- The router has no conversation memory, so purely referential follow-up questions ("explain that differently") can be misrouted.
- Citation page numbers depend on the PDF having extractable page-level text; scanned/image-only PDFs won't produce accurate page metadata.

## Project structure

```
agentic-rag-research-assistant/
├── main.py                  # Inngest functions: ingestion + the full query pipeline
├── router.py                 # Routing step - decides if retrieval is needed
├── retriever.py               # Qdrant search + (legacy) score-based confidence check
├── confidence_judge.py         # LLM-based relevance check, replaces score threshold
├── data_loader.py              # PDF chunking with page-number tracking
├── vector_db.py                 # Qdrant wrapper (upsert, search, list known sources)
├── custom_types.py               # Pydantic models for every pipeline stage
├── streamlit_app.py                # Upload + query UI
├── eval/
│   ├── eval_set.json                # 16 labeled test questions
│   └── run_eval.py                   # Evaluation harness
└── uploads/                            # Local PDF storage (gitignored)
```

## Author

Raphael Obomhese — Fullstack & Mobile Engineer
[Portfolio](https://portfolioorm.vercel.app) · [GitHub](https://github.com/Obomhese-Raphael) · [LinkedIn](https://linkedin.com/in/obomheser)
