web: gunicorn frontend:app \
  --worker-class gthread \
  --workers 1 \
  --threads 4 \
  --timeout 600 \
  --graceful-timeout 120 \
  --max-requests 200 \
  --max-requests-jitter 50 \
  --log-level info \
  --bind 0.0.0.0:$PORT