# Starting the CDCP AI API

## Quick Start

```bash
# Navigate to the api-py directory
cd apps/api-py

# Start the API server
python -m src.main
```

The API will start on: **http://localhost:8001**

---

## Available Endpoints

Once running, access:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **Health Check**: http://localhost:8001/health
- **RAG Health**: http://localhost:8001/rag/health

---

## Test the API

### 1. Health Check

```bash
curl http://localhost:8001/rag/health
```

Expected response:
```json
{
  "status": "healthy",
  "embedding_service": "loaded",
  "vector_db": "loaded",
  "total_documents": 0
}
```

### 2. Ingest CDCP Content

```bash
curl -X POST "http://localhost:8001/rag/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "base_urls": ["https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"],
    "max_pages": 5,
    "allowed_paths": ["/dental-care-plan/"]
  }'
```

### 3. Search

```bash
curl -X POST "http://localhost:8001/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is eligible for CDCP?",
    "n_results": 3
  }'
```

### 4. Check Stats

```bash
curl http://localhost:8001/rag/stats
```

---

## Troubleshooting

### Port Already in Use

If port 8001 is in use, change it:

```bash
PORT=8002 python -m src.main
```

Or edit `main.py` line 68:
```python
port = int(os.getenv("PORT", 8002))  # Changed from 8001
```

### Import Errors

Make sure you're running from the `api-py` directory:
```bash
cd apps/api-py
python -m src.main
```

### Missing Dependencies

Install required packages:
```bash
pip install fastapi uvicorn chromadb transformers torch beautifulsoup4 requests tiktoken pysbd
```

---

## Production Deployment

For production, use uvicorn directly:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

Or with gunicorn:

```bash
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```
