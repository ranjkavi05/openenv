# ── Build stage ──────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Nginx)
RUN apt-get update && apt-get install -y \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose Hugging Face public port
EXPOSE 7860

# Copy project files
COPY . .

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Create needed directories and set permissions for non-root execution
RUN mkdir -p /tmp/client_temp /tmp/proxy_temp /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp && \
    chmod -R 777 /tmp

# Make sure start.sh is executable
RUN chmod +x start.sh

# Run the startup script
ENTRYPOINT ["./start.sh"]
