# File: partner_backend/Dockerfile

# 1. Use Python 3.11 slim image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (for caching speed)
COPY requirements.txt .

# 4. Install dependencies
# We add --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Start the app
# "app:app" means: look in file 'app.py' for the object 'app'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]