#!/bin/bash
# Start the Streamlit application in the background (Port 8501)
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &

# Start the FastAPI server in the background (Port 8000)
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Nginx in the foreground (Port 7860)
nginx -c /etc/nginx/nginx.conf -g "daemon off;"
