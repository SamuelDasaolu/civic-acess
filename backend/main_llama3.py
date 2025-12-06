
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from rag_engine import RAGEngine
# Use the modern, correct client for Hugging Face Inference API
from huggingface_hub import InferenceClient

# --- Startup Check for HF_TOKEN ---
hf_token_check = os.getenv("HF_TOKEN")
if hf_token_check:
    print("--- HF_TOKEN found in environment. ---")
else:
    print("--- FATAL: HF_TOKEN not found in environment. Please check .idx/dev.nix ---")
# -------------------------------------

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API is running. Ready to receive requests at /chat."}

# --- RAG Engine Initialization ---
try:
    print("--- Initializing RAG Engine from main.py ---")
    rag_engine = RAGEngine()
    rag_engine.load_constitution()
    print("--- RAG Engine initialized successfully ---")
except Exception as e:
    print(f"--- FATAL ERROR: RAG Engine failed to initialize: {e} ---")
    raise

# --- New Inference Client Initialization ---
# With the 'unstable' channel, the library is new enough to support this.
# We explicitly set the base_url to force the use of the correct router.
try:
    print("--- Initializing Hugging Face Inference Client ---")
    inference_client = InferenceClient(base_url="https://router.huggingface.co")
    print("--- Hugging Face Inference Client initialized successfully. ---")
except Exception as e:
    print(f"--- FATAL ERROR: Inference Client failed to initialize: {e} ---")
    raise

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    user_question = chat_message.message
    
    print("--- Step 1: Querying RAG Engine ---")
    legal_context = rag_engine.query_law(user_question)
    
    # --- Define Prompts and Model ---
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    system_prompt = "You are a Nigerian legal assistant. Your job is to simplify complex legal text from the Nigerian constitution into plain English or Pidgin for a layperson. Use the provided legal context to answer the user's question."
    user_prompt = f"Based on the context below, answer the user's question.\n\n[Legal Context]\n{legal_context}\n\n[User Question]\n{user_question}"

    # The error message confirmed this model requires a conversational task.
    # The new library version supports the .chat.completions.create() method.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"--- Step 2: Calling LLM ({model_id}) via chat.completions.create ---")
    try:
        # Use the modern, OpenAI-compatible method as required by the API.
        response = inference_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=500,
        )
        
        # Extract the content from the response
        generated_text = response.choices[0].message.content
        
        print("--- Step 3: Successfully received response from LLM ---")
        return {"generated_text": generated_text}

    except Exception as e:
        # Provide a more detailed error message
        error_detail = {
            "message": "Failed to get a successful response from the Hugging Face InferenceClient.",
            "model_id": model_id,
            "error_type": type(e).__name__,
            "error_details": str(e)
        }
        print(f"--- ERROR: {error_detail} ---")
        raise HTTPException(status_code=500, detail=error_detail)
