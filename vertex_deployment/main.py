import os
import uvicorn
import logging
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- AGGRESSIVE LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vertex-logger")

app = FastAPI()

# 1. Configuration
MODEL_ID = "NCAIR1/N-ATLaS"
HF_TOKEN = os.getenv("HF_TOKEN")

model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    logger.info("--- 🚀 STARTING SERVER: INITIALIZING MODEL LOADING ---")
    
    try:
        logger.info(f"--- Loading Tokenizer for {MODEL_ID}... ---")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        
        logger.info("--- Loading Model (16-bit Full Precision)... ---")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True
        )
        logger.info("--- ✅ MODEL LOADED SUCCESSFULLY ON GPU ---")
        
    except Exception as e:
        # We print the FULL traceback to logs so you can see exactly why it failed
        logger.exception("--- ❌ FATAL ERROR DURING MODEL LOAD ---")

@app.get("/health")
def health():
    # Vertex AI pings this. Return 200 even if loading so we don't get killed.
    if model:
        return {"status": "healthy", "gpu": torch.cuda.get_device_name(0)}
    else:
        return {"status": "loading_or_failed"}

@app.post("/predict")
async def predict(request: Request):
    if not model:
        logger.error("Request received but model is not loaded.")
        return {"error": "Model failed to load. Check logs."}

    try:
        body = await request.json()
        instances = body.get("instances", [])
        if not instances: 
            return {"error": "No instances found in request body"}
        
        prompt = instances[0].get("prompt")
        logger.info(f"--- Received Prompt (Length: {len(prompt)}) ---")

        # Inference
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("--- Inference Successful ---")
        
        return {"predictions": [result]}

    except Exception as e:
        logger.exception("--- ❌ ERROR DURING INFERENCE ---")
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)