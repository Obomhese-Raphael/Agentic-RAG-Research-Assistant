import asyncio
import json
import time
from pathlib import Path

import requests
import inngest

INNGEST_API_BASE = "http://127.0.0.1:8288/v1"


async def send_query_event(question: str, top_k: int = 5) -> str:
    client = inngest.Inngest(app_id="agentic-rag-app", is_production=False)
    ids = await client.send(
        inngest.Event(name="rag/query_pdf_ai",
                      data={"question": question, "top_k": top_k})
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
                raise RuntimeError(f"Run {status}")

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                output = run.get("output")
                if output:
                    return output

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out (last status: {last_status})")
        time.sleep(0.5)


def load_eval_set() -> list[dict]:
    eval_path = Path(__file__).parent / "eval_set.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_eval():
    eval_set = load_eval_set()
    results = []

    routing_correct = 0
    routing_total = 0

    print(f"Running {len(eval_set)} eval questions...\n")

    for item in eval_set:
        question = item["question"]
        expected_retrieval = item["expects_retrieval"]
        expected_relevant = item.get("expects_relevant_match")

        event_id = await send_query_event(question)
        output = wait_for_run_output(event_id)

        actual_retrieval = output.get("used_retrieval", False)
        top_score = output.get("top_score")
        confidence = output.get("confidence")

        routing_match = (actual_retrieval == expected_retrieval)
        routing_total += 1
        if routing_match:
            routing_correct += 1

        results.append({
            "id": item["id"],
            "question": question,
            "expected_retrieval": expected_retrieval,
            "actual_retrieval": actual_retrieval,
            "routing_correct": routing_match,
            "expected_relevant_match": expected_relevant,
            "top_score": top_score,
            "confidence": confidence,
        })

        status = "OK" if routing_match else "MISROUTED"
        print(f"[{status}] Q{item['id']}: {question}")
        print(f"    expected_retrieval={expected_retrieval} actual={actual_retrieval} "
              f"top_score={top_score} confidence={confidence}")

    print("\n" + "=" * 70)
    print(f"ROUTING ACCURACY: {routing_correct}/{routing_total} "
          f"({100 * routing_correct / routing_total:.1f}%)")
    print("=" * 70)

    # Separate the relevant-match vs irrelevant-match scores so we can
    # eyeball where a clean threshold would actually sit.
    relevant_scores = [r["top_score"] for r in results
                       if r["expected_relevant_match"] is True and r["top_score"] is not None]
    irrelevant_scores = [r["top_score"] for r in results
                         if r["expected_relevant_match"] is False and r["top_score"] is not None]

    print("\nTop scores for EXPECTED RELEVANT matches:")
    print(sorted(relevant_scores))
    print("\nTop scores for EXPECTED IRRELEVANT matches (should ideally be lower):")
    print(sorted(irrelevant_scores))

    if relevant_scores and irrelevant_scores:
        print(f"\nLowest relevant score:    {min(relevant_scores):.4f}")
        print(f"Highest irrelevant score: {max(irrelevant_scores):.4f}")
        if min(relevant_scores) > max(irrelevant_scores):
            suggested = (min(relevant_scores) + max(irrelevant_scores)) / 2
            print(
                f"Clean separation exists. Suggested threshold: {suggested:.4f}")
        else:
            print("WARNING: scores overlap - no single threshold cleanly separates "
                  "relevant from irrelevant matches with this embedding model on this corpus.")

    # Save full results to a file for the README / writeup
    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_eval())
