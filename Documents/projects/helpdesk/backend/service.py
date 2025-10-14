import os
from datetime import datetime
import logging
import time
from backend import models
from backend import database
from ai.ai_pipeline import full_ticket_analysis
from backend.integrations import zendesk, servicenow, freshdesk
from sqlalchemy.orm import Session
from tests.debug_logger import trace_function, log_debug


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Create ticket with immediate commit
@trace_function()
def create_ticket(db: Session, ticket_create):
    """Create ticket with immediate commit."""
    db_ticket = models.Ticket(
        message=ticket_create.message,
        category="PENDING",  # Changed from UNKNOWN to PENDING
        subcategory="none",
        analyzed=False,
        created_at=datetime.now()
    )
    db.add(db_ticket)
    db.commit()
    
    db.refresh(db_ticket)
    log_debug(f"✅ Ticket {db_ticket.id} created successfully")
    return db_ticket

# Background analysis: create a fresh DB session here (BackgroundTasks / worker safe)
@trace_function()
def analyze_ticket_service(ticket_id: int):
    print(f"[Background Task] STARTING analysis for ticket {ticket_id}")
    
    max_retries = 2
    retry_delay = 2
    
    for attempt in range(max_retries + 1):
        db = None
        try:
            print(f"[Background Task] Attempt {attempt + 1} for ticket {ticket_id}")
            
            db = database.SessionLocal()
            ticket = db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
            
            if not ticket:
                print(f"[Background Task] Ticket {ticket_id} not found")
                return

            print(f"[Background Task] Processing: {ticket.message[:100]}...")
            
            # AI analysis
            print(f"[Background Task] Calling AI pipeline...")
            result = full_ticket_analysis(ticket.message)
            print(f"[Background Task] AI pipeline returned: {result.get('category')}")

            # Update ticket
            ticket.category = result.get("category", "OTHER")
            ticket.subcategory = result.get("subcategory", "general")
            ticket.summary = result.get("summary", "No summary")
            ticket.response = result.get("response", "No response")
            ticket.analyzed = True
            ticket.updated_at = datetime.now()

            db.commit()
            print(f"[Background Task] SUCCESS - Ticket {ticket_id} analyzed as {ticket.category}")
            return
            
        except Exception as e:
            print(f"[Background Task] Attempt {attempt + 1} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if db:
                db.rollback()
            
            if attempt < max_retries:
                print(f"[Background Task] Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"[Background Task] All retries failed for ticket {ticket_id}")
                _create_fallback_ticket(ticket_id)
                return
        finally:
            if db:
                db.close()

# Fallback analysis if AI fails completely

def _create_fallback_ticket(ticket_id: int):
    db = None
    try:
        db = database.SessionLocal()
        ticket = db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
        if ticket:
            category = _infer_category_fallback(ticket.message)
            ticket.category = category
            ticket.subcategory = "general"
            ticket.summary = f"User reported: {ticket.message[:80]}..."
            ticket.response = _get_fallback_response(category)
            ticket.analyzed = True
            ticket.updated_at = datetime.now()
            db.commit()
            log_debug(f"🔄 Fallback analysis for ticket {ticket_id}: {category}")
    except Exception as e:
        log_debug(f"💥 Even fallback failed for ticket {ticket_id}: {e}")
    finally:
        if db:
            db.close()

# Simple keyword-based fallback classification

def _infer_category_fallback(text: str) -> str:
    text_lower = text.lower()
    if any(word in text_lower for word in ["login", "password", "account", "email"]):
        return "ACCOUNT"
    elif any(word in text_lower for word in ["order", "delivery", "shipping", "package"]):
        return "ORDER" 
    elif any(word in text_lower for word in ["charge", "payment", "bill", "refund"]):
        return "BILLING"
    elif any(word in text_lower for word in ["subscription", "cancel", "renew"]):
        return "SUBSCRIPTION"
    elif any(word in text_lower for word in ["crash", "error", "technical", "slow"]):
        return "TECHNICAL"
    return "OTHER"

# Simple fallback responses

def _get_fallback_response(category: str) -> str:
    responses = {
        "ACCOUNT": "I understand you're having account issues. Let me help you resolve this.",
        "ORDER": "I see you have an order-related concern. Let me look into this for you.",
        "BILLING": "I understand your billing concern. Let me check this for you.",
        "SUBSCRIPTION": "I can help with your subscription question.", 
        "TECHNICAL": "I understand you're experiencing technical difficulties.",
        "OTHER": "Thank you for your message. I'll help you with this."
    }
    return responses.get(category, "Thank you for your message. We'll assist you shortly.")

def get_ticket(db, ticket_id: int):
    return db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()

def get_tickets(db):
    return db.query(models.Ticket).order_by(models.Ticket.created_at.desc()).all()


# Background push: create a fresh DB session here (BackgroundTasks / worker safe)
def push_to_integrations(ticket_id: int):
    db = database.SessionLocal()
    try:
        ticket = get_ticket(db, ticket_id)
        if not ticket:
            logger.error("Ticket %s not found for pushing", ticket_id)
            return

        external_ids = ticket.external_ids or {}

        # Zendesk
        if os.getenv("ZENDESK_ENABLED", "false").lower() == "true":
            try:
                z_id = zendesk.create_ticket_in_zendesk(ticket)
                external_ids["zendesk"] = z_id
                logger.info("Pushed ticket %s to Zendesk id=%s", ticket.id, z_id)
            except Exception as e:
                logger.exception("Zendesk push failed: %s", e)

        # ServiceNow
        if os.getenv("SERVICENOW_ENABLED", "false").lower() == "true":
            try:
                sn_id = servicenow.create_incident(ticket)
                external_ids["servicenow"] = sn_id
                logger.info("Pushed ticket %s to ServiceNow id=%s", ticket.id, sn_id)
            except Exception as e:
                logger.exception("ServiceNow push failed: %s", e)

        # Freshdesk
        if os.getenv("FRESHDESK_ENABLED", "false").lower() == "true":
            try:
                fd_id = freshdesk.create_ticket(ticket)
                external_ids["freshdesk"] = fd_id
                logger.info("Pushed ticket %s to Freshdesk id=%s", ticket.id, fd_id)
            except Exception as e:
                logger.exception("Freshdesk push failed: %s", e)

        ticket.external_ids = external_ids
        db.commit()
        db.refresh(ticket)
    finally:
        db.close()
