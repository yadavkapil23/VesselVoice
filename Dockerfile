# Stage 1: Build React Frontend
FROM node:18-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python Backend
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies for librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files and requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy prediction logic and models
COPY predict.py .
COPY server.py .
COPY "Background and motivation/" "Background and motivation/"
COPY "Classification models/" "Classification models/"
COPY "Feature extraction/" "Feature extraction/"
COPY "Performance Evaluation/" "Performance Evaluation/"
COPY Preprocessing/ Preprocessing/
COPY data/ data/

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
