# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artefacts
COPY models/ ./models/

# Copy FastAPI application directly into /app
COPY app/main.py ./main.py

# Expose port 80
EXPOSE 80

# Start the API — main.py is directly in /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]