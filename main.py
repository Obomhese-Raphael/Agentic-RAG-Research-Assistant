import logging
import os
import uuid
import datetime
from fastapi import FastAPI
from dotenv import load_dotenv

import inngest
import inngest.fast_api
from inngest.experimental import ai
from llama_index.core import Settings

from data_loader import get_chunks_and_pages
from vector_db import QdrantStorage
from router import route_question
from retriever import retrieve, check_confidence, format_sources
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGQueryResult,
    RouteDecision,
    RAGSearchResult,
    ConfidenceLevel,
)

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="agentic-rag-app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="Agentic RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id")

        chunks, pages = get_chunks_and_pages(pdf_path)

        return RAGChunkAndSrc(chunks=chunks, source_id=source_id, pages=pages)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        pages = chunks_and_src.pages or [None] * len(chunks)
        source_id = chunks_and_src.source_id

        vectors = Settings.embed_model.get_text_embedding_batch(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}"))
            for i in range(len(vectors))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i], "page": pages[i]}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids=ids, vector=vectors, payload=payloads)

        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load_and_chunk_pdf",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        "embed_and_upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="Agentic RAG: Query",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data.get("question")
    top_k = int(ctx.event.data.get("top_k", 5))

    # STEP 1: Routing - decide if retrieval is even needed.
    # We tell the router which documents are actually loaded so it
    # doesn't skip retrieval just because the model "happens to know"
    # a famous text - the answer should still be grounded in the PDF.
    known_sources = await ctx.step.run(
        "get_known_sources",
        lambda: QdrantStorage().get_known_sources()
    )

    route_decision = await ctx.step.run(
        "route_question",
        lambda: route_question(question, known_sources),
        output_type=RouteDecision
    )

    adapter = ai.openai.Adapter(
        auth_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b",
    )

    # CASE A: no retrieval needed - answer directly, no documents involved
    if not route_decision.needs_retrieval:
        res = await ctx.step.ai.infer(
            "generate_answer_no_retrieval",
            adapter=adapter,
            body={
                "max_tokens": 512,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
            }
        )
        answer = res["choices"][0]["message"]["content"].strip()

        result = RAGQueryResult(
            answer=answer,
            used_retrieval=False,
            confidence=None,
            sources=[]
        )
        return result.model_dump()

    # CASE B: retrieval needed
    search_result = await ctx.step.run(
        "embed_and_search",
        lambda: retrieve(question, top_k),
        output_type=RAGSearchResult
    )

    confidence = await ctx.step.run(
        "check_confidence",
        lambda: check_confidence(search_result),
        output_type=ConfidenceLevel
    )

    sources = format_sources(search_result)
    context_block = "\n\n".join(f" - {c.text}" for c in search_result.chunks)

    # If confidence is low, tell the model explicitly so it can hedge
    # in its own wording, rather than answering as if it's sure.
    if confidence.level == "low":
        caveat_instruction = (
            "IMPORTANT: The retrieved context below may not be very relevant "
            "to the question (low similarity match). If you cannot find a clear "
            "answer in the context, say so explicitly and warn the user that "
            "your answer may be incomplete or unreliable, rather than guessing."
        )
    else:
        caveat_instruction = ""

    user_content = (
        f"{caveat_instruction}\n\n"
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer based on the context provided. If the context does not actually "
        "address the question, do not just say 'I don't know' - instead, briefly "
        "explain that the document does not cover this topic, and say a sentence "
        "or two about what the document actually does cover instead, so the user "
        "understands why their question can't be answered from it."
    )

    res = await ctx.step.ai.infer(
        "generate_answer_with_retrieval",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for answering questions based on provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )
    answer = res["choices"][0]["message"]["content"].strip()

    result = RAGQueryResult(
        answer=answer,
        used_retrieval=True,
        confidence=confidence.level,
        top_score=confidence.top_score,
        sources=sources
    )
    return result.model_dump()


app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
