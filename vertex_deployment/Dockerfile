# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to be sent straight to the terminal
# without being first buffered.
ENV PYTHONUNBUFFERED True

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Set the Hugging Face token as a build argument
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the app.
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app -k uvicorn.workers.UvicornWorker
