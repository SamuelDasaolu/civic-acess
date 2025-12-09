FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install utilities
RUN apt-get update && apt-get install -y git curl

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy App
COPY main.py .

# Environment
ENV AIP_HTTP_PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]