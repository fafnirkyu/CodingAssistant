import os
import requests
from dotenv import load_dotenv

load_dotenv()

ZENDESK_DOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_TOKEN = os.getenv("ZENDESK_TOKEN")

BASE_URL = f"https://{ZENDESK_DOMAIN}.zendesk.com/api/v2"

auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_TOKEN)
headers = {"Content-Type": "application/json"}

def get_new_tickets(status="new"):
    url = f"{BASE_URL}/search.json?query=type:ticket status:{status}"
    r = requests.get(url, auth=auth)
    r.raise_for_status()
    return r.json().get("results", [])

def add_internal_note(ticket_id: int, body: str):
    url = f"{BASE_URL}/tickets/{ticket_id}.json"
    payload = {
        "ticket": {
            "comment": {"body": body, "public": False}
        }
    }
    r = requests.put(url, json=payload, auth=auth, headers=headers)
    r.raise_for_status()
    return r.json()

def reply_to_user(ticket_id: int, body: str):
    url = f"{BASE_URL}/tickets/{ticket_id}.json"
    payload = {
        "ticket": {
            "comment": {"body": body, "public": True}
        }
    }
    r = requests.put(url, json=payload, auth=auth, headers=headers)
    r.raise_for_status()
    return r.json()
