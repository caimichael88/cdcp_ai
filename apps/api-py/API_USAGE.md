# CDCP AI - API Usage Guide

## API Endpoints

Base URL: `http://localhost:8000/rag`

---

## 1. Ingest Content (Scrape & Store)

**POST** `/rag/ingest`

Scrape CDCP pages, chunk, embed, and store in vector database.

### Request Body

```json
{
  "base_urls": [
    "https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"
  ],
  "max_pages": 10,
  "allowed_paths": ["/dental-care-plan/"],
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

### Response

```json
{
  "success": true,
  "scraped_documents": 8,
  "chunked_documents": 45,
  "ingested_documents": 45,
  "scraping_time": 26.5,
  "message": "Successfully ingested 8 documents (45 chunks)"
}
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/rag/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "base_urls": ["https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"],
    "max_pages": 10,
    "allowed_paths": ["/dental-care-plan/"]
  }'
```

---

## 2. Search Documents

**POST** `/rag/search`

Semantic search across stored documents.

### Request Body

```json
{
  "query": "Who is eligible for CDCP?",
  "n_results": 5,
  "filter_section": "eligibility"
}
```

### Response

```json
{
  "query": "Who is eligible for CDCP?",
  "results": [
    {
      "content": "To qualify for the CDCP, you must be a Canadian resident...",
      "title": "Canadian Dental Care Plan - Do you qualify",
      "url": "https://www.canada.ca/.../qualify.html",
      "section": "eligibility",
      "similarity_score": 0.89
    },
    {
      "content": "Eligibility requirements include...",
      "title": "CDCP Eligibility Guide",
      "url": "https://www.canada.ca/.../guide.html",
      "section": "eligibility",
      "similarity_score": 0.85
    }
  ],
  "total_results": 2
}
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is eligible for CDCP?",
    "n_results": 5
  }'
```

---

## 3. Get Statistics

**GET** `/rag/stats`

Get database statistics.

### Response

```json
{
  "total_documents": 45,
  "collection_name": "cdcp_documents"
}
```

### cURL Example

```bash
curl -X GET "http://localhost:8000/rag/stats"
```

---

## 4. Health Check

**GET** `/rag/health`

Check if services are loaded and ready.

### Response

```json
{
  "status": "healthy",
  "embedding_service": "loaded",
  "vector_db": "loaded",
  "total_documents": 45
}
```

### cURL Example

```bash
curl -X GET "http://localhost:8000/rag/health"
```

---

## 5. Clear Database

**DELETE** `/rag/clear`

Clear all documents from the database (use with caution!).

### Response

```json
{
  "status": "success",
  "message": "Database cleared"
}
```

### cURL Example

```bash
curl -X DELETE "http://localhost:8000/rag/clear"
```

---

## Complete Workflow Example

### Step 1: Ingest CDCP Pages

```bash
# Scrape and store CDCP documentation
curl -X POST "http://localhost:8000/rag/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "base_urls": [
      "https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"
    ],
    "max_pages": 20,
    "allowed_paths": ["/dental-care-plan/"],
    "chunk_size": 512,
    "chunk_overlap": 50
  }'
```

### Step 2: Check Statistics

```bash
# Verify ingestion
curl -X GET "http://localhost:8000/rag/stats"
```

### Step 3: Search

```bash
# Search for information
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I apply for dental coverage?",
    "n_results": 3
  }'
```

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/rag"

# 1. Ingest content
response = requests.post(f"{BASE_URL}/ingest", json={
    "base_urls": [
        "https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"
    ],
    "max_pages": 10,
    "allowed_paths": ["/dental-care-plan/"]
})
print(response.json())

# 2. Search
response = requests.post(f"{BASE_URL}/search", json={
    "query": "Who is eligible for CDCP?",
    "n_results": 5
})
results = response.json()

for i, result in enumerate(results['results']):
    print(f"\nResult {i+1}:")
    print(f"Title: {result['title']}")
    print(f"Score: {result['similarity_score']:.2f}")
    print(f"Content: {result['content'][:200]}...")

# 3. Get stats
response = requests.get(f"{BASE_URL}/stats")
print(response.json())
```

---

## JavaScript/TypeScript Client Example

```typescript
const BASE_URL = 'http://localhost:8000/rag';

// 1. Ingest content
const ingestResponse = await fetch(`${BASE_URL}/ingest`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    base_urls: [
      'https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html'
    ],
    max_pages: 10,
    allowed_paths: ['/dental-care-plan/']
  })
});
const ingestData = await ingestResponse.json();
console.log(ingestData);

// 2. Search
const searchResponse = await fetch(`${BASE_URL}/search`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Who is eligible for CDCP?',
    n_results: 5
  })
});
const searchData = await searchResponse.json();

searchData.results.forEach((result, i) => {
  console.log(`\nResult ${i+1}:`);
  console.log(`Title: ${result.title}`);
  console.log(`Score: ${result.similarity_score}`);
  console.log(`Content: ${result.content.substring(0, 200)}...`);
});
```

---

## Error Responses

### 404 - No Documents Found

```json
{
  "detail": "No documents were scraped"
}
```

### 500 - Internal Server Error

```json
{
  "detail": "Error message here"
}
```

### 503 - Service Unavailable

```json
{
  "detail": "Service initialization failed"
}
```

---

## Best Practices

1. **Ingestion**:
   - Start with `max_pages: 10` for testing
   - Use `allowed_paths` to stay within relevant content
   - Monitor `scraping_time` to estimate full ingestion duration

2. **Search**:
   - Use `filter_section` to narrow results
   - Adjust `n_results` based on your use case (3-10 recommended)
   - Check `similarity_score` (> 0.7 is usually relevant)

3. **Performance**:
   - First request loads models (slower)
   - Subsequent requests are much faster
   - Consider warming up services with `/health` endpoint

4. **Deployment**:
   - Set appropriate `max_pages` limits in production
   - Monitor database size
   - Implement rate limiting for public APIs

---

## Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation and testing interfaces.
