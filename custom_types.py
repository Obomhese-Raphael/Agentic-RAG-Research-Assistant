import pydantic
from typing import Optional, Literal


# ---------- Ingestion types (same idea as your old project) ----------

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: Optional[str] = None
    # page number per chunk, same order as chunks
    pages: Optional[list[int]] = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


# ---------- NEW: Routing types ----------

class RouteDecision(pydantic.BaseModel):
    needs_retrieval: bool
    reason: str  # short explanation of why the model chose this


# ---------- Retrieval + citation types ----------

class RetrievedChunk(pydantic.BaseModel):
    text: str
    source: str
    page: Optional[int] = None
    score: float  # similarity score from Qdrant, used for confidence


class RAGSearchResult(pydantic.BaseModel):
    chunks: list[RetrievedChunk]
    top_score: float  # highest score among retrieved chunks, used for confidence check


# ---------- NEW: Confidence types ----------

class ConfidenceLevel(pydantic.BaseModel):
    level: Literal["high", "low"]
    top_score: float
    threshold: float


# ---------- Final answer type ----------

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    used_retrieval: bool
    confidence: Optional[Literal["high", "low"]] = None
    top_score: Optional[float] = None
    sources: list[str] = []  # formatted citations, e.g. "report.pdf p.4"
