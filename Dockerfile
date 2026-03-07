# Use official Python 3.10 lightweight image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system libraries required by FAISS and numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache this layer
# (if requirements don't change, Docker won't reinstall packages on rebuild)
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Command that runs when the container starts
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
