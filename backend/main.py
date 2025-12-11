from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine

# --- RATE LIMITING ---
from fastapi import Request 
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- GOOGLE SDKs for TRANSLATE and VERTEX AI ---
from google.cloud import translate_v2 as translate
from google.cloud import aiplatform

# --- IMPORT EVALUATOR AND DB ---
from evaluator import init_db, log_request, lazy_judge, get_db_connection

load_dotenv()

# Initialize Limiter (Tracks users by IP address)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

# Register the Limiter with FastAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# --- Initialize Vertex AI ---
vertex_endpoint = None
try:
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION")
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID")

    if project_id and location and endpoint_id:
        print(f"--- Connecting to Vertex AI Endpoint: {endpoint_id} ---")
        aiplatform.init(project=project_id, location=location)
        vertex_endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
        print("--- Vertex AI Connected Successfully ---")
    else:
        print("--- WARNING: Vertex AI credentials missing in .env ---")
except Exception as e:
    print(f"--- FATAL VERTEX ERROR: {e} ---")

# --- Initialize Translation Client ---
try:
    translate_client = translate.Client()
    print("--- Translation Client Ready ---")
except Exception as e:
    print(f"--- WARNING: Translation Client Failed: {e} ---")
    translate_client = None

class UserQuery(BaseModel):
    message: str = "What is the most supreme law in Nigeria?"
    language: str = "english"

@app.post("/chat")
@limiter.limit("5/minute")
async def chat(query: UserQuery, background_tasks: BackgroundTasks, request: Request):
    user_text = query.message
    target_lang = query.language.lower().strip()

    print(f"--- INCOMING: '{user_text}' -> '{target_lang}' ---")

    # 0. TRANSLATION LAYER 
    search_query = user_text
    detected_lang = "en"
    translation_status = "skipped_for_english"

    if translate_client and target_lang != "english":
        try:
            # Auto-detect and translate to English
            # Result is a dict: {'input': '...', 'translatedText': '...', 'detectedSourceLanguage': 'fr'}
            result = translate_client.translate(user_text, target_language="en")
            # Check if we actually got a result
            if isinstance(result, dict) and "translatedText" in result:
                search_query = result["translatedText"]
                detected_lang = result["detectedSourceLanguage"]
                translation_status = "success"
                print(f"--- Translated ({detected_lang}): '{user_text}' -> '{search_query}' ---")
            
            # Simple optimization: If user typed in English but selected 'Yoruba' in UI, 
            # we don't need to change anything.
            if detected_lang == "en":
                search_query = user_text

        except Exception as e:
            print(f"--- ⚠️ TRANSLATION FAILED (Quota/Error): {e} ---")
            print("--- Falling back to original user text ---")
            search_query = user_text
            translation_status = "failed_fallback"

    # 1. RAG Lookup
    rag_context = ""
    if rag_engine:
        try:
            context_list = rag_engine.query_law(search_query)
            if context_list:
                rag_context = "\n".join(context_list) if isinstance(context_list, list) else str(context_list)
        except Exception as e:
            rag_context = f"Error: {str(e)}"
    
    # Fallback if RAG is empty
    if not rag_context:
        rag_context = "No specific legal section found."

    # 2. PUPPETEER STRATEGY (Preserved)
    ai_starter = "" 

    if target_lang == "pidgin":
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

    # 4. Call Brain (Using Google Cloud SDK)
    final_answer = "Error: Could not connect to AI Brain."
    
    if vertex_endpoint:
        try:
            print("--- Sending request to Vertex AI SDK... ---")
            # Vertex SDK Call
            response = vertex_endpoint.predict(
                instances=[{"prompt": full_prompt}],
                parameters={"maxOutputTokens": 256, "temperature": 0.5}
            )
            
            # response.predictions is a list. vertex container returns [text].
            if response.predictions:
                raw_reply = response.predictions[0]
                
                # --- CLEANUP LOGIC ---
                if ai_starter in raw_reply:
                    clean_reply = raw_reply.split(ai_starter)[-1] 
                    final_answer = ai_starter + clean_reply
                elif "assistant<|end_header_id|>" in raw_reply:
                    final_answer = raw_reply.split("assistant<|end_header_id|>")[-1].strip()
                else:
                    final_answer = raw_reply
            else:
                final_answer = "Vertex AI returned no predictions."
                
        except Exception as e:
            print(f"Vertex SDK Error: {e}")
            final_answer = f"Vertex Error: {str(e)}"
    else:
        final_answer = "Vertex Endpoint Not Initialized."

    # 5. ASYNC EVALUATION (The 'Lazy Judge')
    try:
        row_id = log_request(f"{user_text} [Translation in English: {search_query}]", target_lang, rag_context, final_answer)
        background_tasks.add_task(lazy_judge, row_id, user_text, rag_context, final_answer)
        print(f"--- Evaluator Triggered for Row {row_id} ---")
    except Exception as e:
        print(f"Evaluator Error: {e}")

    # 6. Return to User
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


@app.get("/logs")
@limiter.limit("5/minute")
def view_logs(request: Request):
    """Fetches logs from Turso Remote DB"""
    try:
        # Use the connection helper from evaluator.py
        conn = get_db_connection()
        if not conn:
            return {"error": "Could not connect to Turso database."}
        
        # Execute query
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        
        # Convert tuples to dictionary manually (Remote drivers vary on row_factory support)
        # We get column names from description
        columns = [description[0] for description in cursor.description]
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
            
        return {"logs": results}
    except Exception as e:
        print(f"LOGS ENDPOINT ERROR: {e}")
        return {"error": f"Internal Error: {str(e)}"}


@app.post("/test-rag")
@limiter.limit("5/minute")
async def test_rag_retrieval(query: UserQuery, request: Request):
    """Test endpoint to see exactly what the RAG engine retrieves."""
    print('RAG TESTING ENDPOINT REACHED')
    user_text = query.message
    target_lang = query.language.lower().strip()

    print(f"--- INCOMING: '{user_text}' -> '{target_lang}' ---")

    # 0. TRANSLATION LAYER 
    search_query = user_text
    detected_lang = "en"

    if translate_client and target_lang != "english":
        try:
            # Auto-detect and translate to English
            # Result is a dict: {'input': '...', 'translatedText': '...', 'detectedSourceLanguage': 'fr'}
            result = translate_client.translate(user_text, target_language="en")
            
            search_query = result["translatedText"]
            detected_lang = result["detectedSourceLanguage"]
            
            print(f"--- Translated ({detected_lang}): '{user_text}' -> '{search_query}' ---")
            
            # Simple optimization: If user typed in English but selected 'Yoruba' in UI, 
            # we don't need to change anything.
            if detected_lang == "en":
                search_query = user_text

        except Exception as e:
            print(f"Translation Error: {e}")

    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized")

    try:
        results = rag_engine.query_law(search_query)
        return {
            "raw_query": query.message,
            "translated_query": search_query,
            "detected_lang": detected_lang,
            "retrieved_chunks": results,
            "chunk_count": len(results) if isinstance(results, list) else 1
        }
    except Exception as e:
        return {"error": str(e)}
    
   
