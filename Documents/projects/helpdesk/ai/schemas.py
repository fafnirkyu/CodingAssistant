from pydantic import BaseModel
from typing import Optional

class HelpdeskTicket(BaseModel):
    category: str
    subcategory: Optional[str] = "none"
    summary: str
    response: str
