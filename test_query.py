import asyncio
import time
import requests
import inngest

INNGEST_API_BASE = "http://127.0.0.1:8288/v1"


async def send_query_event(question: str, top_k: int = 5) -> str:
    client = inngest.Inngest(app_id="agentic-rag-app", is_production=False)

    ids = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    return ids[0]


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{INNGEST_API_BASE}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Run {status}. Full run data: {run}")

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                output = run.get("output")
                if output:
                    return output

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out (last status: {last_status})")
        time.sleep(0.5)


async def main():
    test_questions = [
        "What does Sun Tzu say about knowing your enemy?",
        "What is the capital of France?",
        "What does Sun Tzu say about using cyberattacks and computer networks?",
        "According to the document, what is Sun Tzu's opinion on quantum physics?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        event_id = await send_query_event(q)
        output = wait_for_run_output(event_id)
        print(f"  used_retrieval = {output.get('used_retrieval')}")
        print(f"  confidence     = {output.get('confidence')}")
        print(f"  top_score      = {output.get('top_score')}")
        print(f"  sources        = {output.get('sources')}")
        print(f"  answer         = {output.get('answer')}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
