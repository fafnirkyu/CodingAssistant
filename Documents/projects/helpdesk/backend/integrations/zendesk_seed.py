import os
import requests
from dotenv import load_dotenv

# ✅ Point to backend/.enviorment explicitly
env_path = os.path.join(os.path.dirname(__file__), "..", ".enviorment")
load_dotenv(dotenv_path=env_path)

ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_TOKEN = os.getenv("ZENDESK_API_TOKEN")

if not all([ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_TOKEN]):
    print("❌ Missing Zendesk credentials. Check your .enviorment file.")
    print(f"  SUBDOMAIN={ZENDESK_SUBDOMAIN}")
    print(f"  EMAIL={ZENDESK_EMAIL}")
    print(f"  TOKEN={'✅' if ZENDESK_TOKEN else '❌'}")
    exit(1)

auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_TOKEN)
BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"

# 🧠 Example realistic tickets
SAMPLE_TICKETS = [
    {"subject": "Charged twice on my credit card", "description": "I was charged twice for my purchase on March 15th."},
    {"subject": "App keeps crashing", "description": "The mobile app crashes every time I try to open it on my Android device."},
    {"subject": "Order hasn’t arrived", "description": "My order #45678 was supposed to arrive last week but it hasn’t yet."},
    {"subject": "Can’t log in", "description": "I reset my password but still can’t log into my account."},
    {"subject": "Need invoice", "description": "Can I get an invoice for my last purchase?"},
    {"subject": "Cancel subscription", "description": "I want to cancel my premium subscription before it renews."},
    {"subject": "Refund not processed", "description": "My refund from last month’s order still hasn’t shown up."},
    {"subject": "Promo code invalid", "description": "Tried using SAVE20 but it says invalid promo code."},
    {"subject": "Website slow", "description": "The checkout page takes forever to load on Chrome."},
    {"subject": "Account locked", "description": "My account was locked after several login attempts."},
]

def create_ticket(subject, description):
    """Create a new ticket in Zendesk."""
    payload = {
        "ticket": {
            "subject": subject,
            "comment": {"body": description},
            "priority": "normal"
        }
    }
    resp = requests.post(
        f"{BASE_URL}/tickets.json",
        auth=auth,
        json=payload,
        timeout=20
    )
    if resp.status_code == 201:
        data = resp.json()
        print(f"✅ Created ticket #{data['ticket']['id']} - {subject}")
    else:
        print(f"⚠️ Failed to create ticket: {resp.status_code} | {resp.text}")

def seed_tickets():
    print("🌱 Seeding Zendesk with sample tickets...")
    for t in SAMPLE_TICKETS:
        create_ticket(t["subject"], t["description"])

if __name__ == "__main__":
    seed_tickets()
