# 
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from rag_engine import RAGEngine
# 
# # --- NEW: Import transformers for local inference ---
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch
# # ----------------------------------------------------
# 
# # --- Startup Check for HF_TOKEN ---
# # Still required by transformers to download gated models.
# hf_token = os.getenv("HF_TOKEN")
# if hf_token:
#     print("--- HF_TOKEN found in environment. ---")
# else:
#     print("--- WARNING: HF_TOKEN not found. This may cause issues downloading the model. ---")
# # ------------------------------------
# 
# app = FastAPI()
# 
# @app.get("/")
# async def root():
#     return {"message": "API is running. Ready to receive requests at /chat."}
# 
# # --- RAG Engine Initialization ---
# try:
#     print("--- Initializing RAG Engine from main.py ---")
#     rag_engine = RAGEngine()
#     rag_engine.load_constitution()
#     print("--- RAG Engine initialized successfully. ---\n")
# except Exception as e:
#     print(f"--- FATAL ERROR: RAG Engine failed to initialize: {e} ---")
#     raise
# 
# # --- NEW: Local Model and Pipeline Initialization ---
# model_id = "NCAIR1/N-ATLaS"
# try:
#     print(f"--- Initializing local transformers pipeline for {model_id} ---")
#     print("--- NOTE: The first run will download the model, which may take several minutes. ---")
#     
#     # Load the tokenizer and model from Hugging Face.
#     # The token is necessary to access this gated model.
#     tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, 
#         token=hf_token, 
#         torch_dtype=torch.float16 # Use float16 for memory efficiency
#     )
#     
#     # Create a text-generation pipeline using the loaded model and tokenizer.
#     llm_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         torch_dtype=torch.float16,
#         device_map="auto" # Automatically uses GPU if available, else CPU
#     )
#     
#     print(f"--- Local pipeline for {model_id} initialized successfully. ---\n")
# 
# except Exception as e:
#     print(f"--- FATAL ERROR: Local model pipeline failed to initialize: {e} ---")
#     print("--- Please ensure you have accepted the model's terms on Hugging Face and have a valid HF_TOKEN in .idx/dev.nix ---")
#     raise
# # ----------------------------------------------------
# 
# class ChatMessage(BaseModel):
#     message: str
# 
# @app.post("/chat")
# async def chat(chat_message: ChatMessage):
#     user_question = chat_message.message
#     
#     print("--- Step 1: Querying RAG Engine ---")
#     legal_context = rag_engine.query_law(user_question)
#     
#     # --- Define Prompt for Local Model ---
#     prompt = f"You are a Nigerian legal assistant. Use the provided legal context to answer the user's question.\n\n[Legal Context]\n{legal_context}\n\n[User Question]\n{user_question}"
# 
#     print(f"--- Step 2: Generating text with local LLM ({model_id}) ---")
#     try:
#         # --- NEW: Use the local pipeline for inference ---
#         sequences = llm_pipeline(
#             prompt,
#             max_new_tokens=500,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id
#         )
#         
#         generated_text = sequences[0]['generated_text']
#         
#         print("--- Step 3: Successfully received response from local LLM ---\n")
#         return {"generated_text": generated_text}
# 
#     except Exception as e:
#         error_detail = {
#             "message": "Failed to get a successful response from the local transformers pipeline.",
#             "model_id": model_id,
#             "error_type": type(e).__name__,
#             "error_details": str(e)
#         }
#         print(f"--- ERROR: {error_detail} ---")
#         raise HTTPException(status_code=500, detail=error_detail)
# 