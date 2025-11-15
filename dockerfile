FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch (no CUDA)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
    torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

EXPOSE 5001
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5001"]

