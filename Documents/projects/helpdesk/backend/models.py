from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy import JSON
from sqlalchemy.sql import func
from backend.database import Base

class Ticket(Base):
    __tablename__ = "tickets"

    subject = Column(String, nullable=True)
    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    category = Column(String, default="PENDING")  # Changed from UNKNOWN to PENDING
    subcategory = Column(String, default="none")
    response = Column(Text, nullable=True)
    analyzed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True)