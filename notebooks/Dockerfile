# Base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artefacts
COPY models/ ./models/

# Copy FastAPI application
COPY app/main.py .

# Expose port 80
EXPOSE 80

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]