import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from custom_types import RouteDecision

load_dotenv()

# Raw OpenAI-compatible client pointed at Groq - same pattern as your
# inference call in main.py, just used here for a much smaller/cheaper job.
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

ROUTER_SYSTEM_PROMPT_TEMPLATE = """You are a routing assistant. Your only job is to decide \
whether answering the user's question requires searching a document database, \
or whether it can be answered directly from general knowledge / conversation \
context alone.

The following documents are currently loaded and searchable: {document_list}

Respond with retrieval needed if the question:
- Could plausibly be answered using one of the loaded documents above, even \
if you already know something about the general topic from training data. \
If a loaded document covers this subject, prefer retrieval so the answer is \
grounded in that specific document rather than general knowledge.
- References "the document", "the paper", "the report", "this PDF", etc.
- Asks something specific enough that a generic answer would likely be wrong

Respond with no retrieval needed if the question:
- Is a greeting, small talk, or about the conversation itself \
(e.g. "what did I just ask you?")
- Is a general knowledge question clearly unrelated to any loaded document's topic
- Is a request to clarify or rephrase something already answered

Respond ONLY with valid JSON, no other text, in this exact shape:
{{"needs_retrieval": true or false, "reason": "short one-sentence reason"}}
"""


def route_question(question: str, known_sources: list[str] | None = None) -> RouteDecision:
    if known_sources:
        document_list = ", ".join(known_sources)
    else:
        document_list = "(none currently loaded)"

    system_prompt = ROUTER_SYSTEM_PROMPT_TEMPLATE.format(
        document_list=document_list)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=150,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    raw_content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw_content)
        return RouteDecision(
            needs_retrieval=bool(parsed.get("needs_retrieval", True)),
            reason=parsed.get("reason", "no reason given")
        )
    except (json.JSONDecodeError, AttributeError):
        # If the model ever returns something that isn't valid JSON,
        # fail safe: default to "needs retrieval" rather than skipping
        # search and risking a made-up answer.
        return RouteDecision(
            needs_retrieval=True,
            reason="router output could not be parsed, defaulting to retrieval"
        )
