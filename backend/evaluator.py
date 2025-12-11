import os
import json
import libsql
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# --- CONFIGURATION ---
TURSO_URL = os.getenv("TURSO_URL")
TURSO_TOKEN = os.getenv("TURSO_TOKEN")
JUDGE_MODEL = "gemini-2.5-flash-preview-09-2025"

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_db_connection():
    """Establishes a connection to the Turso remote database."""
    if not TURSO_URL or not TURSO_TOKEN:
        print("--- Error: TURSO_URL or TURSO_TOKEN missing in .env ---")
        return None
        
    try:
        # Connect to Turso using the sync client
        conn = libsql.connect(TURSO_URL, auth_token=TURSO_TOKEN)
        return conn
    except Exception as e:
        print(f"--- Turso Connection Error: {e} ---")
        return None

def init_db():
    """Creates the table in Turso if it doesn't exist."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        # Note: We use conn.execute() directly for libsql
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
        print("--- Database Initialized on Turso ---")
    except Exception as e:
        print(f"--- DB Init Error: {e} ---")

def log_request(query, lang, context, reply):
    """Logs the chat interaction to Turso."""
    conn = get_db_connection()
    if not conn: return None
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interactions (timestamp, user_query, target_lang, rag_context, model_reply)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, query, lang, context, reply))
        conn.commit()
        
        # Get the ID of the row we just inserted
        # Note: lastrowid availability depends on the driver version, 
        # but usually works for standard inserts.
        return cursor.lastrowid
    except Exception as e:
        print(f"--- Logging Error: {e} ---")
        return None

def lazy_judge(row_id, user_query, rag_context, model_reply):
    """
    Uses Gemini to grade the response and updates the row in Turso.
    """
    if not row_id:
        return

    print(f"--- [Judge] Grading Row {row_id}... ---")
    
    if not api_key:
        print("❌ [Judge] Error: No GEMINI_API_KEY found.")
        return

    try:
        # 1. Initialize Model
        model = genai.GenerativeModel(JUDGE_MODEL)
        
        # 2. Prompt
        prompt = f"""
        Act as an impartial legal evaluator. 
        Compare the AI's Response against the Reference Legal Context.

        Query: "{user_query}"
        Reference Context: "{rag_context}"
        AI Response: "{model_reply}"

        Evaluation Criteria:
        1. Accuracy (0-100): Does the AI response strictly follow the Reference Context?
        2. Hallucination: Did the AI invent facts not in the Reference?
        3. Clarity: Is the answer simple?
        4. Language: Is the AI response in any of the target languages (English, Yoruba, Hausa or Wazobia Pidgin)?
        5. It is however imperative that the model completes it's reply in a single language, the only acceptable code-switching is english/pidgin.
        

        Output Format:
        Return a valid JSON string ONLY. Do not use Markdown.
        {{
            "score": 85,
            "reason": "The explanation matches the context perfectly."
        }}
        """
        
        # 3. Call API
        response = model.generate_content(prompt)
        text = response.text
        
        # 4. Cleaning & Parsing
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        start = clean_text.find("{")
        end = clean_text.rfind("}") + 1
        if start != -1 and end != -1:
            clean_text = clean_text[start:end]

        try:
            data = json.loads(clean_text)
            score = data.get("score", 0)
            reason = data.get("reason", "No reason provided")
        except json.JSONDecodeError:
            score = 0
            reason = f"JSON Parse Failed. Raw: {clean_text[:30]}..."

        # 5. Update Turso DB
        conn = get_db_connection()
        if conn:
            conn.execute('''
                UPDATE interactions 
                SET judge_score = ?, judge_reason = ?, status = 'graded'
                WHERE id = ?
            ''', (score, reason, row_id))
            conn.commit()
            print(f"--- [Judge] Success: Score {score} ---")

    except Exception as e:
        print(f"❌ [Judge] Error: {e}")