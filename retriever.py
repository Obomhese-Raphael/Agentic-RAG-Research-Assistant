from llama_index.core import Settings

from vector_db import QdrantStorage
from custom_types import RAGSearchResult, ConfidenceLevel, RetrievedChunk

# Below this similarity score, we treat retrieval as "weak" - this number
# is a starting guess. The eval harness (eval/run_eval.py) is what will
# actually tell us if 0.65 is too strict or too loose for this corpus.
CONFIDENCE_THRESHOLD = 0.65


def retrieve(question: str, top_k: int = 5) -> RAGSearchResult:
    """
    Embeds the question, searches Qdrant, and returns the retrieved
    chunks plus the single highest similarity score among them.
    """
    query_vector = Settings.embed_model.get_text_embedding(question)

    store = QdrantStorage()
    chunks: list[RetrievedChunk] = store.search(
        query_vector=query_vector, top_k=top_k)

    if chunks:
        top_score = max(chunk.score for chunk in chunks)
    else:
        top_score = 0.0

    return RAGSearchResult(chunks=chunks, top_score=top_score)


def check_confidence(search_result: RAGSearchResult, threshold: float = CONFIDENCE_THRESHOLD) -> ConfidenceLevel:
    """
    Compares the top retrieved score against a threshold to decide
    whether this retrieval should be trusted or flagged as weak.
    """
    level = "high" if search_result.top_score >= threshold else "low"

    return ConfidenceLevel(
        level=level,
        top_score=search_result.top_score,
        threshold=threshold
    )


def format_sources(search_result: RAGSearchResult) -> list[str]:
    """
    Turns the raw retrieved chunks into a clean, deduplicated list of
    human-readable citations, e.g. "report.pdf p.4".
    """
    seen = set()
    formatted = []

    for chunk in search_result.chunks:
        if chunk.page is not None:
            citation = f"{chunk.source} p.{chunk.page}"
        else:
            citation = chunk.source

        if citation not in seen:
            seen.add(citation)
            formatted.append(citation)

    return formatted
