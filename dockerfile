FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: expose logs directory
VOLUME ["/app/logs"]

EXPOSE 5001

# Run your API script (assuming it starts the FastAPI/Flask server)
CMD ["python", "api.py"]
