# Use lightweight official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install minimal build tools if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the REQUIRED application frontend and backend files
COPY app/ /app/app/
COPY ALL_13_MODELS/ML_models/ /app/ALL_13_MODELS/ML_models/
COPY tactical_ml_v4.joblib /app/tactical_ml_v4.joblib
COPY main.py /app/main.py

# Create a non-root user for security (Hugging Face standard)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860

# Launch Uvicorn with the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
