import os
import sys

# ✅ Ensure project root (helpdesk/) is in Python's path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from ai.ai_pipeline import full_ticket_analysis
from backend import database, models


# Load environment variables
load_dotenv("backend/.enviorment")

ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_TOKEN = os.getenv("ZENDESK_API_TOKEN")

if not all([ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_TOKEN]):
    raise ValueError("❌ Missing Zendesk credentials. Check your .enviorment file.")

auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_TOKEN)
BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"

PROCESSED_IDS = set()

def fetch_new_tickets():
    """Fetch Zendesk tickets."""
    try:
        resp = requests.get(f"{BASE_URL}/tickets.json?page[size]=25", auth=auth, timeout=20)
        resp.raise_for_status()
        return resp.json().get("tickets", [])
    except Exception as e:
        print(f"⚠️ Failed to fetch tickets: {e}")
        return []

def post_comment(ticket_id: int, message: str):
    """Add an AI-generated public comment to a Zendesk ticket."""
    payload = {"ticket": {"comment": {"body": message, "public": True}}}
    try:
        resp = requests.put(f"{BASE_URL}/tickets/{ticket_id}.json", auth=auth, json=payload, timeout=20)
        if resp.status_code == 200:
            print(f"💬 Posted AI response to Ticket #{ticket_id}")
        else:
            print(f"⚠️ Failed to post comment: {resp.status_code} | {resp.text}")
    except Exception as e:
        print(f"⚠️ Error posting comment: {e}")

def save_ticket_to_db(ticket_id, subject, message, category, summary, response):
    """Save ticket to the local helpdesk.db."""
    db: Session = database.SessionLocal()
    try:
        ticket = models.Ticket(
            id=ticket_id,
            subject=subject,
            message=message,
            category=category,
            summary=summary,
            response=response,
            analyzed=True
        )
        db.merge(ticket)  # merge ensures it updates if it already exists
        db.commit()
        print(f"💾 Saved Ticket #{ticket_id} → DB")
    except Exception as e:
        print(f"⚠️ Failed to save ticket #{ticket_id} to DB: {e}")
    finally:
        db.close()

def process_ticket(ticket):
    ticket_id = ticket["id"]
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")
    
    print(f"\n🎟️ Processing Ticket #{ticket_id}: {subject}")
    analysis = full_ticket_analysis(description)
    
    category = analysis.get("category", "OTHER")
    summary = analysis.get("summary", "")
    response = analysis.get("response", "")
    
    print(f"✅ AI Analysis: {category} | {summary}")
    save_ticket_to_db(ticket_id, subject, description, category, summary, response)
    post_comment(ticket_id, f"[AI Helpdesk] {response}")

def run_automation_loop(interval=60):
    """Continuously fetch and process new Zendesk tickets."""
    print("🤖 Starting Zendesk Auto Processor... (Ctrl+C to stop)")
    while True:
        tickets = fetch_new_tickets()
        for t in tickets:
            if t["id"] not in PROCESSED_IDS:
                process_ticket(t)
                PROCESSED_IDS.add(t["id"])
        time.sleep(interval)

if __name__ == "__main__":
    run_automation_loop(interval=30)
