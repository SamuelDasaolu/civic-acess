import sqlite3
import os
import google.generativeai as genai
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "chat_logs.db"

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env. Judge will fail.")

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
    Background Task: Uses Gemini 1.5 Flash to grade the answer.
    """
    print(f"--- [Judge] STARTING for Row {row_id}... ---")
    
    # --- CRITICAL ERROR CATCHING ---
    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("Missing GEMINI_API_KEY in environment variables")

        # 1. Initialize the Judge
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 2. The Grading Prompt
        prompt = f"""
        Act as an impartial legal expert. Evaluate the accuracy of the AI's response.

        **Input Data:**
        - User Question: "{user_query}"
        - Truth (Legal Context): "{rag_context}"
        - AI Response: "{model_reply}"

        **Criteria:**
        1. Accuracy (0-100): Does the response correctly reflect the Legal Context?
        2. Hallucination: Did the AI invent laws not in the text?
        3. Clarity: Is the translation/explanation simple?

        **Output Format:**
        Provide JSON ONLY with these keys: "score" (int) and "reason" (string).
        """
        
        # 3. Call Gemini
        print(f"--- [Judge] Calling Google Gemini API... ---")
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        
        # 4. Parse Result
        try:
            data = json.loads(text)
            score = data.get("score", 0)
            reason = data.get("reason", "No reason provided")
        except:
            score = 0
            reason = "Failed to parse Judge output JSON"

        # 5. Update Database
        print(f"--- [Judge] Updating DB for Row {row_id} with Score {score}... ---")
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE interactions 
            SET judge_score = ?, judge_reason = ?, status = 'graded'
            WHERE id = ?
        ''', (score, reason, row_id))
        conn.commit()
        conn.close()
        
        print(f"--- [Judge] SUCCESS: Row {row_id} Graded. ---")

    except Exception as e:
        # THIS IS WHAT YOU NEED TO SEE
        print(f"❌ [Judge] CRASHED: {str(e)}")
        
        # Log the error to DB so you see it in /logs endpoint too
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE interactions 
                SET status = ? 
                WHERE id = ?
            ''', (f"ERROR: {str(e)}", row_id))
            conn.commit()
            conn.close()
        except:
            pass