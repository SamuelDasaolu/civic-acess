import os
import libsql
from dotenv import load_dotenv
from typing import Generator
from fastapi import HTTPException
import time

load_dotenv()

# --- CONFIGURATION ---
TURSO_URL = os.getenv("TURSO_URL")
TURSO_TOKEN = os.getenv("TURSO_TOKEN")

_db_connection = None
_last_sync = 0
SYNC_INTERVAL = 5  # Sync every 5 seconds, not on every request


def get_db_connection() -> libsql.Connection | None:
    """
    Reuses a single connection instead of creating a new one each time.
    """
    global _db_connection, _last_sync
    
    if not TURSO_URL or not TURSO_TOKEN:
        print("--- Error: TURSO_URL or TURSO_TOKEN missing in .env ---")
        return None
    
    try:
        # If no connection exists, create one
        if _db_connection is None:
            print("Creating new Turso connection...")
            _db_connection = libsql.connect(
                database="local.db",  
                sync_url=TURSO_URL,
                auth_token=TURSO_TOKEN
            )
            _db_connection.sync()
            _last_sync = time.time()
        
        # Only sync if enough time has passed
        current_time = time.time()
        if current_time - _last_sync > SYNC_INTERVAL:
            print("Syncing with remote Turso...")
            _db_connection.sync()
            _last_sync = current_time
        
        return _db_connection
    except Exception as e:
        print(f"--- Turso Connection Error: {e} ---")
        _db_connection = None
        return None


def init_db():
    """Creates tables in the database if they don't exist."""
    conn = get_db_connection()
    if not conn:
        print("--- Cannot initialize DB: Connection failed. ---")
        return

    try:
        # Create user table for authentication
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create interactions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_query TEXT,
                target_lang TEXT,
                rag_context TEXT,
                model_reply TEXT,
                judge_score INTEGER,
                judge_reason TEXT,
                status TEXT DEFAULT 'pending'
            )
        ''')
        conn.commit()
        conn.sync()  # Sync changes to remote
        print("✅ Database Initialized on Turso (user and interactions tables)")
    except Exception as e:
        print(f"--- DB Init Error: {e} ---")


# --- FastAPI Dependency ---

def get_db() -> Generator[libsql.Connection, None, None]:
    """
    Dependency function that provides a reused database connection.
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    try:
        yield conn
    except Exception as e:
        print(f"Database error: {e}")
        raise