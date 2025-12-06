
import os
import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# A dictionary to hold the model and tokenizer once loaded
ml_models = {}

# This is the lifespan context manager. It will run the code inside
# on application startup and teardown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP --- #
    # This code runs when the server starts.
    print("INFO:     Application startup: Loading model...")

    # It's crucial to have the HF_TOKEN set in the environment.
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set. Please provide your Hugging Face token.")

    # Determine the device (GPU or CPU) and print it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO:     Using device: {device}")

    # Load the tokenizer and the model from Hugging Face into the ml_models dictionary.
    # We use the 'token' argument for authentication.
    # trust_remote_code=True is required for this specific model.
    ml_models["tokenizer"] = AutoTokenizer.from_pretrained("NCAIR1/N-ATLaS", token=hf_token)
    ml_models["model"] = AutoModel.from_pretrained("NCAIR1/N-ATLaS", token=hf_token, trust_remote_code=True).to(device)
    
    print("INFO:     Model loading complete.")

    # The 'yield' statement separates the startup code from the shutdown code.
    yield

    # --- SHUTDOWN --- #
    # This code runs when the server is shutting down.
    print("INFO:     Application shutdown: Clearing models from memory.")
    ml_models.clear()

# Initialize the FastAPI app, passing our lifespan manager to it.
app = FastAPI(lifespan=lifespan)

# A health check endpoint that Vertex AI will use to make sure
# the server is running and healthy.
@app.get("/health", status_code=200)
def health():
    return {"status": "ok"}

# The main prediction endpoint.
@app.post("/predict")
async def predict(request: Request):
    """
    Accepts a JSON payload with a "text" field, sends it to the model,
    and returns the resulting embedding vector.
    """
    try:
        body = await request.json()
        text = body.get("text")
    except Exception as e:
        return {"error": f"Invalid request body: {e}"}, 400

    if not text or not isinstance(text, str):
        return {"error": "'text' field must be a non-empty string"}, 400

    # Retrieve the tokenizer and model from our dictionary.
    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]
    device = model.device

    # Tokenize the input text and move the resulting tensors to the GPU/CPU.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Run the model inference. torch.no_grad() is a performance optimization.
    with torch.no_grad():
        outputs = model(**inputs)
        # This model stores its embedding in 'last_hidden_state'.
        # We then average the token embeddings to get a single sentence embedding.
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        # Move the final tensor to the CPU and convert it to a standard Python list.
        response_data = embedding.cpu().numpy().tolist()

    return {"embedding": response_data}

