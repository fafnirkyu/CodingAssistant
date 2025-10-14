import json

def log_misclassified(ticket_text: str, expected: str, predicted: str):
    """Store misclassified tickets for future fine-tuning."""
    entry = {"text": ticket_text, "expected": expected, "predicted": predicted}
    with open("misclassified.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")