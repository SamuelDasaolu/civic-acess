from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import re
from dotenv import load_dotenv
from rag_engine import RAGEngine

# --- IMPORT EVALUATOR ---
import sqlite3
from evaluator import init_db, log_request, lazy_judge 


load_dotenv()

app = FastAPI()

# --- CORS FIX ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize DB on Startup ---
init_db()

# --- Initialize RAG ---
try:
    print("--- Initializing RAG Engine ---")
    rag_engine = RAGEngine()
    rag_engine.load_constitution()
    print("--- RAG Engine initialized successfully. ---\n")
except Exception as e:
    print(f"FATAL RAG ERROR: {e}")
    rag_engine = None

class UserQuery(BaseModel):
    message: str
    language: str = "english"

@app.post("/chat")
async def chat(query: UserQuery, background_tasks: BackgroundTasks):
    user_text = query.message
    target_lang = query.language.lower().strip()

    print(f"--- INCOMING: '{user_text}' -> '{target_lang}' ---")

    # 1. RAG Lookup
    rag_context = ""
    if rag_engine:
        try:
            context_list = rag_engine.query_law(user_text)
            if context_list:
                rag_context = "\n".join(context_list) if isinstance(context_list, list) else str(context_list)
        except Exception as e:
            rag_context = f"Error: {str(e)}"
    
    # Fallback if RAG is empty
    if not rag_context:
        rag_context = "No specific legal section found."

    # 2. PUPPETEER STRATEGY (The Final Logic)
    ai_starter = "" 

    if target_lang == "pidgin":
        # Force "Street Pidgin" to avoid Yoruba Pivot
        system_instruction = """Act like a street guy from Lagos. 
Translate the [Legal Context] into pure Nigerian Pidgin English.
Do NOT use Yoruba words (like 'naa', 'ni', 'wipe').
Use 'na', 'dey', 'we', 'dem'."""
        ai_starter = "My guy, dis law talk say"

    elif target_lang == "yoruba":
        system_instruction = """Translate the main idea of the [Legal Context] into very simple Yoruba.
Do not use big legal words."""
        ai_starter = "Ofin yii sọ ni ṣókí pé"

    elif target_lang == "hausa":
        system_instruction = """Translate the main idea of the [Legal Context] into very simple Hausa."""
        ai_starter = "Wannan dokar ta ce"

    elif target_lang == "igbo":
        # Inject vocabulary to help the model
        system_instruction = """Translate the main idea of the [Legal Context] into simple Igbo.
Use 'Usoro Iwu' for Constitution.
Use 'kachasị elu' for Supreme."""
        ai_starter = "Usoro Iwu a kwuru na"

    else: # English
        system_instruction = """You are a Nigerian legal assistant. 
Explain the [Legal Context] simply."""
        ai_starter = "Basically, the law states that"

    # 3. Construct Prompt (Llama-3 Format)
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_instruction}

[Legal Context]
{rag_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{ai_starter}"""

    # 4. Call Brain
    BRAIN_URL = os.getenv("MODEL_API_URL")
    final_answer = "Error: Could not connect to AI Brain."
    
    if BRAIN_URL:
        try:
            payload = {
                "prompt": full_prompt,
                "max_new_tokens": 256
            }
            
            response = requests.post(BRAIN_URL, json=payload, timeout=45)
            
            if response.status_code == 200:
                raw_reply = response.json().get("generated_text", "")
                
                # --- CLEANUP LOGIC ---
                if ai_starter in raw_reply:
                    # Robust Split: Keep starter + rest
                    clean_reply = raw_reply.split(ai_starter)[-1] 
                    final_answer = ai_starter + clean_reply
                elif "assistant<|end_header_id|>" in raw_reply:
                    final_answer = raw_reply.split("assistant<|end_header_id|>")[-1].strip()
                else:
                    final_answer = raw_reply
            else:
                final_answer = f"Brain Error {response.status_code}: {response.text}"
        except Exception as e:
            final_answer = f"Connection Error: {str(e)}"
    
    # 5. ASYNC EVALUATION (The 'Lazy Judge')
    try:
        # Save pending record
        row_id = log_request(user_text, target_lang, rag_context, final_answer)
        
        # Trigger background grading
        background_tasks.add_task(lazy_judge, row_id, user_text, rag_context, final_answer)
        print(f"--- Evaluator Triggered for Row {row_id} ---")
    except Exception as e:
        print(f"Evaluator Error: {e}")

    # 6. Return to User
    return {
        "response": final_answer,
        "debug_info": {
            "language": target_lang,
            "strategy": "Puppeteer V2 + Lazy Judge",
            "rag_context": rag_context
        }
    }


@app.get("/logs")
def view_logs():
    """See the graded answers with Error Handling"""
    db_path = "chat_logs.db"
    
    # Check if file exists
    if not os.path.exists(db_path):
        return {"error": f"Database file '{db_path}' not found. Have you chatted with the bot yet?"}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'")
        if not cursor.fetchone():
            conn.close()
            return {"error": "Database exists but table 'interactions' is missing. Check init_db() in evaluator.py"}

        # Fetch Logs
        cursor.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()
        
        return {"logs": [dict(row) for row in rows]}

    except Exception as e:
        # This will print the REAL error to your terminal and the browser
        print(f"LOGS ENDPOINT ERROR: {e}")
        return {"error": f"Internal Error: {str(e)}"}