import asyncio
from pathlib import Path
import inngest

# Change this to the actual filename you placed in uploads/
PDF_FILENAME = "THE ART OF WAR.pdf"


async def main():
    client = inngest.Inngest(app_id="agentic-rag-app", is_production=False)

    pdf_path = Path("uploads") / PDF_FILENAME

    if not pdf_path.exists():
        print(f"File not found: {pdf_path.resolve()}")
        return

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    print(f"Ingestion event sent for: {pdf_path.name}")
    print("Check the Inngest dev server UI (usually http://127.0.0.1:8288) to watch it run.")


if __name__ == "__main__":
    asyncio.run(main())
