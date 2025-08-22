FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trading inference engine
COPY ../inference_container/trading_inference_engine.py .

# Copy the API server script
COPY app.py .

# Expose port for API
EXPOSE 8503

# Run the API server
CMD ["python", "app.py"]
