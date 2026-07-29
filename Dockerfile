# Use a lightweight Python 3.11 base image
FROM python:3.11-slim

# Install system dependencies (gcc, gfortran) needed to build scikit-learn
RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Railway will use
EXPOSE 8000

# Run the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
