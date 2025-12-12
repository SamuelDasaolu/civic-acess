from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine

# --- RATE LIMITING ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- GOOGLE SDKs ---
from google.cloud import translate_v2 as translate
import google.generativeai as genai

# --- DATABASE & EVALUATOR ---
from evaluator import init_db, log_request, lazy_judge, get_db_connection

load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ CRITICAL: GEMINI_API_KEY missing. App will crash on generation.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INIT DB & RAG ---
init_db()

try:
    print("--- Initializing RAG Engine ---")
    rag_engine = RAGEngine()
    rag_engine.load_constitution()
    print("--- RAG Engine initialized successfully. ---\n")
except Exception as e:
    print(f"FATAL RAG ERROR: {e}")
    rag_engine = None

# --- INIT TRANSLATION ---
try:
    translate_client = translate.Client()
    print("--- Translation Client Ready ---")
except Exception as e:
    print(f"--- WARNING: Translation Client Failed: {e} ---")
    translate_client = None

# --- PROMPT STRATEGY (UPDATED FOR STRICT LANGUAGE ADHERENCE) ---
PERSONA_MAP = {
    "pidgin": """You are a knowledgeable 'street lawyer' from Lagos. 
    Regardless of the user's input language, you MUST answer in pure Nigerian Pidgin English (Waffi/Lagos blend).
    Use words like 'na', 'dey', 'we', 'dem', 'commot'. 
    Make it sound relatable but strictly accurate.""",
    
    "yoruba": """You are a wise legal advisor. 
    Regardless of the user's input language, you MUST answer in clear, simple Yoruba.
    Translate the legal concepts into Yoruba. 
    Avoid archaic proverbs that obscure the meaning; focus on clarity.""",
    
    "hausa": """You are a helpful legal assistant. 
    Regardless of the user's input language, you MUST answer in Hausa. 
    Explain the law simply and clearly in Hausa.""",
    
    "igbo": """You are a legal assistant. 
    Regardless of the user's input language, you MUST answer in Igbo. 
    Translate the concepts into simple Igbo. Use 'Usoro Iwu' for Constitution.""",
    
    "english": """You are a professional Nigerian legal assistant. 
    Answer in simple, plain English (ELI15). Avoid excessive jargon."""
}

SYSTEM_PROMPT_TEMPLATE = """
You are N-Atlas, an expert AI Legal Assistant for Nigeria.

CORE INSTRUCTIONS:
1. **Persona & Language:** {persona_instruction}
2. **Strict Language Enforcement:** You must ALWAYS respond in the target language defined by your persona above. If the user asks in English but your persona is Hausa, you MUST answer in Hausa.
3. **Source of Truth:** Use ONLY the provided [Legal Context] below to answer. 
   - If the answer is found in the context, cite the section (e.g., "According to Section 33...").
   - If the answer is NOT in the context, clearly state: "I don't know the answer to that question based on the provided context" (translated into the target language). Do NOT make up laws.
4. **Tone:** Empathetic, clear, and authoritative on the facts. In escalation scenarios (police/landlord), try to be the de-escalator.

[Legal Context]:
{context}
"""

class UserQuery(BaseModel):
    message: str
    language: str = "english"

@app.post("/chat")
@limiter.limit("10/minute") 
async def chat(query: UserQuery, background_tasks: BackgroundTasks, request: Request):
    user_text = query.message
    target_lang = query.language.lower().strip()

    print(f"--- INCOMING: '{user_text}' -> '{target_lang}' ---")

    # 1. TRANSLATION LAYER (Critical for RAG Accuracy)
    search_query = user_text
    detected_lang = "en"
    translation_status = "skipped_for_english"

    if translate_client and target_lang != "english":
        try:
            # We translate input to English strictly for the RAG search
            # This ensures we find the right law even if asked in Hausa
            result = translate_client.translate(user_text, target_language="en")
            if isinstance(result, dict) and "translatedText" in result:
                search_query = result["translatedText"]
                detected_lang = result["detectedSourceLanguage"]
                translation_status = "success"
                print(f"--- Translated ({detected_lang}): '{user_text}' -> '{search_query}' ---")
            
            if detected_lang == "en": search_query = user_text

        except Exception as e:
            print(f"--- ⚠️ TRANSLATION ERROR: {e} ---")
            search_query = user_text
            translation_status = "failed_fallback"

    # 2. RAG LOOKUP (High Context)
    rag_context = ""
    if rag_engine:
        try:
            # We use 10 chunks to leverage Gemini's large context window
            context_list = rag_engine.query_law(search_query, final_k=10)
            if context_list:
                rag_context = "\n\n".join(context_list)
        except Exception as e:
            print(f"--- ⚠️ RAG ERROR: {e} ---")
            rag_context = "System Error: Could not retrieve legal context."
    
    if not rag_context:
        rag_context = "No specific legal section found relating to this query."

    # 3. GENERATION LAYER (Gemini 2.0 Flash)
    final_answer = ""
    
    try:
        # Get Persona
        persona_instruction = PERSONA_MAP.get(target_lang, PERSONA_MAP["english"])

        # Hydrate System Prompt
        filled_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            persona_instruction=persona_instruction,
            context=rag_context
        )
        
        # Initialize Model
        model = genai.GenerativeModel(
            "gemini-2.0-flash", 
            system_instruction=filled_system_prompt
        )
        
        # Generate
        response = model.generate_content(
            f"USER QUESTION: {user_text}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3, # Low temp = High Accuracy
                max_output_tokens=800
            )
        )
        
        final_answer = response.text.strip()
        
    except Exception as e:
        # --- SECURE ERROR HANDLING ---
        print(f"❌ GENERATION ERROR (Detailed): {e}")
        final_answer = "I am currently experiencing high traffic. Please try again in a moment."

    # 4. EVALUATION
    try:
        row_id = log_request(f"{user_text} [Trans: {search_query}]", target_lang, rag_context, final_answer)
        background_tasks.add_task(lazy_judge, row_id, user_text, rag_context, final_answer)
    except Exception as e:
        print(f"Evaluator Error: {e}")

    return {
        "response": final_answer,
        "debug_info": {
            "language": target_lang,
            "strategy": "Vertex AI SDK + Puppeteer + Lazy Judge",
            "translation_status": translation_status,
            "translated_query": search_query,
            "rag_context": rag_context
        }
    }

# --- ENDPOINTS FOR DEBUGGING ---

@app.get("/logs")
@limiter.limit("5/minute")
def view_logs(request: Request):
    """Fetches logs from Turso so you can demo the 'Evaluation' feature."""
    try:
        conn = get_db_connection()
        if not conn: return {"error": "DB Connection Failed"}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        
        if rows:
            columns = [d[0] for d in cursor.description]
            return {"logs": [dict(zip(columns, row)) for row in rows]}
        return {"logs": []}
        
    except Exception as e: 
        print(f"LOGS ERROR: {e}")
        return {"error": "Could not retrieve logs."}

@app.post("/test-rag")
@limiter.limit("5/minute")
async def test_rag_retrieval(query: UserQuery, request: Request):
    """Isolate the Retrieval Block to check chunks."""
    user_text = query.message
    search_query = user_text
    detected_lang = "en"
    # Simulate Translation
    if translate_client and query.language.lower().strip() != "english":
        try:
            res = translate_client.translate(user_text, target_language="en")
            search_query = res['translatedText']
            detected_lang = res['detectedSourceLanguage']
        except: pass

    if not rag_engine: raise HTTPException(500, "RAG Error")
    
    # Test with High K to see what Gemini *would* see
    results = rag_engine.query_law(search_query, final_k=10) 
    return {
        "raw_query": user_text,
        "translated_query": search_query,
        "detected_lang": detected_lang,
        "retrieved_chunks": results
    }