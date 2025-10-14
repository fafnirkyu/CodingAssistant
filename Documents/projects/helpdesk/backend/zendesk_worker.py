import time
from ai.ai_pipeline import full_ticket_analysis
from integrations.zendesk import get_new_tickets, add_internal_note

def process_zendesk_tickets(interval: int = 60):
    print("Zendesk worker started...")
    seen = set()

    while True:
        tickets = get_new_tickets()
        for t in tickets:
            if t["id"] in seen:
                continue
            seen.add(t["id"])

            message = t.get("description", "")
            print(f"Processing Zendesk ticket #{t['id']}...")

            result = full_ticket_analysis(message)
            note = (
                f"AI Analysis\n"
                f"Category: {result['category']}\n"
                f"Summary: {result['summary']}\n\n"
                f"Suggested Response:\n{result['response']}"
            )

            add_internal_note(t["id"], note)
            print(f"✅ Ticket #{t['id']} analyzed.")
        
        time.sleep(interval)
