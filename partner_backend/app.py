from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import libsql
import os
from dotenv import load_dotenv
from routes.auth import router as auth_router

from database import get_db, init_db

load_dotenv()

# --- App Initialization ---
app = FastAPI(title="CivicAccess Backend")

# Initialize the database when the app starts up
@app.on_event("startup")
def on_startup():
    print("🚀 Starting FastAPI app...")
    # Initialize the DB, creating the table if it doesn't exist
    init_db()


# --- CORS Configuration ---
# You need to define your allowed origins here

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅ CORS configured")

# --- Include the auth router ---
app.include_router(auth_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to CivicAccess API"}

# Example of a route that uses the Turso connection
@app.get("/interactions/count")
def get_interaction_count(conn: Annotated[libsql.Connection, Depends(get_db)]):
    """
    Retrieves the total count of interactions from the Turso database.
    """
    try:
        result = conn.execute("SELECT COUNT(id) AS count FROM interactions").fetchone()
        return {"total_interactions": result['count']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")
