from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import backend.database as database
from backend import schemas, service
from tests.debug_logger import log_debug
import os

load_dotenv(".enviorment")
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_TOKEN = os.getenv("ZENDESK_TOKEN")

app = FastAPI(title="AI Helpdesk Assistant")

@app.on_event("startup")
def on_startup():
    log_debug("FastAPI starting up...")
    # only initialize DB schema here. Do NOT analyze tickets by default.
    database.init_db()

@app.post("/tickets", response_model=schemas.TicketResponse)
def create_ticket(
    ticket: schemas.TicketCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    new_ticket = service.create_ticket(db, ticket)
    # Run the AI analysis asynchronously in background
    background_tasks.add_task(service.analyze_ticket_service, new_ticket.id)
    return new_ticket

@app.post("/ticket/{ticket_id}/analyze_and_push", response_model=schemas.TicketResponse)
def analyze_and_push(ticket_id: int, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    ticket = service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    # schedule analysis + integration push as background tasks
    background_tasks.add_task(service.analyze_ticket_service, ticket.id)
    return ticket

@app.get("/tickets", response_model=list[schemas.TicketResponse])
def list_tickets(db: Session = Depends(database.get_db)):
    return service.get_tickets(db)

@app.get("/ticket/{ticket_id}", response_model=schemas.TicketResponse)
def get_ticket(ticket_id: int, db: Session = Depends(database.get_db)):
    ticket = service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket
