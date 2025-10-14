from pydantic import BaseModel
from datetime import datetime

class TicketCreate(BaseModel):
    message: str

class TicketResponse(BaseModel):
    id: int
    message: str
    summary: str | None
    category: str
    subcategory: str
    response: str | None
    analyzed: bool
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True
