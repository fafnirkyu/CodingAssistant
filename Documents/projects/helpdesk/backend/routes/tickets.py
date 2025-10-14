from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend import schemas, crud, database
from ai import classifier, summarizer, responder

router = APIRouter()

@router.post("/tickets", response_model=schemas.Ticket)
def create_ticket(ticket: schemas.TicketCreate, db: Session = Depends(database.get_db)):
    category = classifier.classify_ticket(ticket.description)
    summary = summarizer.summarize_ticket(ticket.description)
    ai_reply = responder.generate_response(ticket.description)

    return crud.create_ticket(db=db, ticket=ticket, category=category, summary=summary, ai_reply=ai_reply)
