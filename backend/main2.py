from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import re
from dotenv import load_dotenv
from rag_engine import RAGEngine

# import evaluator
from evaluator import init_db, log_request, lazy_judge
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize RAG ---
try:
    print("--- Initializing RAG Engine ---")
    rag_engine = RAGEngine()
    rag_engine.load_constitution()
    print("--- RAG Engine initialized successfully. ---\n")
except Exception as e:
    rag_engine = None

class UserQuery(BaseModel):
    message: str
    language: str = "english"

@app.post("/chat")
async def chat(query: UserQuery):
    user_text = query.message
    target_lang = query.language.lower().strip()

    print(f"--- INCOMING: '{user_text}' -> '{target_lang}' ---")

    # 1. RAG Lookup
    rag_context = "No specific legal section found."
    if rag_engine:
        try:
            context_list = rag_engine.query_law(user_text)
            if context_list:
                rag_context = "\n".join(context_list) if isinstance(context_list, list) else str(context_list)
        except Exception as e:
            rag_context = f"Error: {str(e)}"

    # 2. PUPPETEER STRATEGY (Refined for Failure Modes)
    
    ai_starter = "" 

    if target_lang == "pidgin":
        # FIX: Removed "Omo" to prevent Yoruba pivot.
        # Added negative constraint against Yoruba words.
        system_instruction = """Act like a street guy from Lagos. 
Translate the [Legal Context] into pure Nigerian Pidgin English.
Do NOT use Yoruba words (like 'naa', 'ni', 'wipe').
Use 'na', 'dey', 'we', 'dem'.
"""
        ai_starter = "My guy, dis law talk say" # <--- Changed from 'Omo'

    elif target_lang == "yoruba":
        system_instruction = """Translate the main idea of the [Legal Context] into very simple Yoruba.
Do not use big legal words."""
        ai_starter = "Ofin yii sọ ni ṣókí pé"

    elif target_lang == "hausa":
        system_instruction = """Translate the main idea of the [Legal Context] into very simple Hausa."""
        ai_starter = "Wannan dokar ta ce"

    elif target_lang == "igbo":
        # FIX: Injected vocabulary for Constitution.
        system_instruction = """Translate the main idea of the [Legal Context] into simple Igbo.
Use 'Usoro Iwu' for Constitution.
Use 'kachasị elu' for Supreme."""
        ai_starter = "Usoro Iwu a kwuru na" # <--- Changed starter to use specific term

    else: # English
        system_instruction = """You are a Nigerian legal assistant. 
Explain the [Legal Context] simply."""
        ai_starter = "Basically, the law states that"

    # 3. Construct Prompt
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_instruction}

[Legal Context]
{rag_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{ai_starter}"""

    # 4. Call Brain
    BRAIN_URL = os.getenv("MODEL_API_URL")
    if not BRAIN_URL:
        return {"response": "Error: No Brain URL configured."}

    try:
        # Note: Removed temperature from payload as server might ignore it.
        # We rely on the prompt's strong guidance.
        payload = {
            "prompt": full_prompt,
            "max_new_tokens": 256
        }
        
        response = requests.post(BRAIN_URL, json=payload, timeout=45)
        
        if response.status_code == 200:
            raw_reply = response.json().get("generated_text", "")
            
            # --- CLEANUP LOGIC ---
            if ai_starter in raw_reply:
                # Keep the starter + whatever followed
                clean_reply = raw_reply.split(ai_starter)[-1] 
                final_answer = ai_starter + clean_reply
            else:
                # Fallback if model decided to repeat the prompt weirdly
                if "assistant<|end_header_id|>" in raw_reply:
                    clean_reply = raw_reply.split("assistant<|end_header_id|>")[-1].strip()
                    final_answer = clean_reply
                else:
                    final_answer = raw_reply

            return {
                "response": final_answer,
                "debug_info": {
                    "language": target_lang,
                    "strategy": "Puppeteer V2 (Vocab Injection)",
                    "starter_used": ai_starter
                }
            }
        else:
            return {"response": f"Brain Error {response.status_code}: {response.text}"}

    except Exception as e:
        return {"response": f"Connection Error: {str(e)}"}