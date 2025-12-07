# Use a lightweight PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install git (needed for some hugginface operations)
RUN apt-get update && apt-get install -y git

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY main.py .

# Vertex AI requires port 8080
EXPOSE 8080

# Start the server
CMD ["python", "main.py"]