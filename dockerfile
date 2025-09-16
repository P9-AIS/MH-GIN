# Base image (same as in your devcontainer.json)
FROM mcr.microsoft.com/devcontainers/python:1-3.8-bullseye

# Set working directory
WORKDIR /app

# Copy requirements first for efficient caching
COPY src/requirements.txt /app/src/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Copy the rest of the code
COPY . /app
