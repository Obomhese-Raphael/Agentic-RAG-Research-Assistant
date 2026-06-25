from router import route_question

test_questions = [
    "What does the report say about Q3 revenue?",
    "Hi, how are you?",
    "What is the capital of France?",
    "Summarize the methodology section of the paper.",
    "What did I just ask you?",
]

for q in test_questions:
    decision = route_question(q)
    print(f"Q: {q}")
    print(f"  needs_retrieval = {decision.needs_retrieval}")
    print(f"  reason          = {decision.reason}")
    print()
