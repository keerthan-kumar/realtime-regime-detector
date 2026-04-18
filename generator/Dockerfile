# Lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY simulate.py .

# Run the generator
CMD ["python", "-u", "simulate.py"]