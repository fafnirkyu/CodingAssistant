from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "helpdesk.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# IMPROVED: Much more robust SQLite configuration
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 30,  # 30 second timeout
    },
    pool_size=10,           # Reduced for SQLite
    max_overflow=20,        # Moderate overflow
    pool_timeout=30,        # 30 second timeout
    pool_recycle=3600,      # Recycle hourly
    pool_pre_ping=True,     # Verify connections
    echo=False              # Disable SQL logging for performance
)

# Enable SQLite WAL mode for better concurrency
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
    cursor.execute("PRAGMA synchronous=NORMAL")  # Better performance
    cursor.execute("PRAGMA cache_size=10000")   # Larger cache
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """Creates database tables if they don't exist."""
    from backend import models
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at {DB_PATH}")

def get_db():
    """Dependency that provides a new DB session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()