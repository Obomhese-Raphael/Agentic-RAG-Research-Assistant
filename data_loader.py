import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

load_dotenv()

# Global LlamaIndex settings - same models as your old project
Settings.llm = Groq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROQ_API_KEY")
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5")

Settings.chunk_size = 1000
Settings.chunk_overlap = 200


def process_pdf_for_index(file_path: str):
    """
    Loads a PDF and splits it into chunks ("nodes"), same as before.
    PDFReader loads one Document per page, so each Document's metadata
    already contains a page_label - we use that to tag every chunk with
    the page it came from, which is what makes citation possible later.
    """
    documents = PDFReader().load_data(file=file_path)

    splitter = SentenceSplitter()
    nodes = splitter.get_nodes_from_documents(documents)

    return nodes


def get_chunks_and_pages(file_path: str) -> tuple[list[str], list[int]]:
    """
    Same chunking as process_pdf_for_index, but returns two parallel lists:
    the chunk text, and the page number each chunk came from.

    This is the function main.py should call during ingestion, since it
    needs both pieces to build citation-ready payloads for Qdrant.
    """
    nodes = process_pdf_for_index(file_path)

    chunks = []
    pages = []

    for node in nodes:
        chunks.append(node.text)

        # PDFReader stores the page number under "page_label" in metadata,
        # as a string (e.g. "1", "2"). Fall back to 0 if it's ever missing.
        page_label = node.metadata.get("page_label", "0")
        try:
            page_num = int(page_label)
        except ValueError:
            page_num = 0
        pages.append(page_num)

    return chunks, pages
