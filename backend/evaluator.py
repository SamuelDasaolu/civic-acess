import sqlite3
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

DB_NAME = "db/chat_logs.db"

# --- CONFIGURATION ---
# We use the standard library but target the Flash model
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Fallback to 'gemini-1.5-flash' which is stable on this library
JUDGE_MODEL = "gemini-2.5-flash-preview-09-2025"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
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
    conn.close()

def log_request(query, lang, context, reply):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO interactions (timestamp, user_query, target_lang, rag_context, model_reply)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, query, lang, context, reply))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id

def lazy_judge(row_id, user_query, rag_context, model_reply):
    """
    Uses Gemini 1.5 Flash (via stable lib) to grade response.
    """
    print(f"--- [Judge] Grading Row {row_id}... ---")
    
    if not api_key:
        print("❌ [Judge] Error: No GEMINI_API_KEY found.")
        return

    try:
        # 1. Initialize Model
        model = genai.GenerativeModel(JUDGE_MODEL)
        
        # 2. Prompt (Explicitly asking for JSON string)
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
        
        # 4. Cleaning & Parsing (Manual logic since we lost strict mode)
        # Remove markdown code blocks if the model adds them
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        # Find JSON boundaries just in case
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
            reason = f"JSON Parse Failed. Raw output: {clean_text[:50]}..."

        # 5. Update DB
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE interactions 
            SET judge_score = ?, judge_reason = ?, status = 'graded'
            WHERE id = ?
        ''', (score, reason, row_id))
        conn.commit()
        conn.close()
        
        print(f"--- [Judge] Success: Score {score}. Reason: {reason} ---")

    except Exception as e:
        print(f"❌ [Judge] Error: {e}")