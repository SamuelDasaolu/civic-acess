# Use lightweight Python
FROM python:3.11-slim

# Install system tools
RUN apt-get update && apt-get install -y git

WORKDIR /app

# Copy requirements
COPY requirements.txt .
# Install dependencies (We use the ones we cleaned up earlier)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Cloud Run expects port 8080 by default
ENV PORT=8080

# Command to run the app
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}