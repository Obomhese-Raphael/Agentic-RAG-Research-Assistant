from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from custom_types import RetrievedChunk


class QdrantStorage:
    # NOTE: different collection name from the old project, so the two
    # projects don't mix data inside the same Qdrant instance (you're
    # reusing the same Docker container, "qdrantRagDB").
    def __init__(self, url="http://localhost:6333", collection="agentic_docs", dim=384):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim, distance=Distance.COSINE)
            )

    def upsert(self, ids: list[str], vector: list[list[float]], payload: list[dict]) -> None:
        points = [
            PointStruct(id=ids[i], vector=vector[i], payload=payload[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k
        )

        chunks: list[RetrievedChunk] = []
        for res in results.points:
            payload = getattr(res, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            # may be None if not set during ingestion
            page = payload.get("page")
            score = getattr(res, "score", 0.0)

            if text:
                chunks.append(
                    RetrievedChunk(text=text, source=source,
                                   page=page, score=score)
                )

        return chunks

    def get_known_sources(self) -> list[str]:
        """
        Returns the unique list of source filenames currently stored in
        this collection. Used by the router so it knows what documents
        actually exist before deciding whether a question needs retrieval.
        """
        sources = set()
        next_offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )

            for point in points:
                source = (point.payload or {}).get("source")
                if source:
                    sources.add(source)

            if next_offset is None:
                break

        return list(sources)
