import os
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "NCAIR1/N-ATLaS"

print("--- INITIALIZING FULL PRECISION MODEL ---")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16  # High Quality (16-bit)
    )
    print("--- MODEL LOADED SUCCESSFULLY ---")
except Exception as e:
    print(f"FATAL ERROR: {e}")
    model = None

@app.get("/health")
async def health():
    return {"status": "healthy"} if model else {"status": "unhealthy"}

@app.post("/predict")
async def predict(request: Request):
    if not model:
        return {"error": "Model failed to load."}

    # Handle Vertex AI Input Format
    body = await request.json()
    # Vertex AI sends: {"instances": [{"prompt": "..."}]}
    if "instances" in body:
        prompt = body["instances"][0].get("prompt")
    else:
        prompt = body.get("prompt")

    if not prompt:
        return {"error": "No prompt provided"}

    # Run Inference
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Vertex AI Output Format
    return {"predictions": [generated_text]}

if __name__ == "__main__":
    port = int(os.getenv("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)