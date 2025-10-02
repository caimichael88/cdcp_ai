# Upsert Behavior - Handling Duplicate URLs

## How It Works

When you ingest the same URL multiple times, the system **automatically updates** the existing content instead of creating duplicates.

---

## ID Generation Strategy

Each chunk gets a **deterministic ID** based on:
- `doc_id`: MD5 hash of the URL (always same for same URL)
- `chunk_index`: Position of the chunk in the document

```python
chunk_id = f"{md5(url)}_chunk_{index}"
# Example: "a3f5b2c9d4e6f7a8_chunk_0"
```

---

## Behavior

### First Ingestion
```bash
POST /rag/ingest
{
  "base_urls": ["https://canada.ca/.../qualify.html"],
  "max_pages": 5
}
```

**Result:**
- Scrapes 5 pages
- Creates 30 chunks
- Stores with IDs: `{doc_id}_chunk_0`, `{doc_id}_chunk_1`, ..., `{doc_id}_chunk_29`

**Database:** 30 documents

---

### Second Ingestion (Same URL)
```bash
POST /rag/ingest
{
  "base_urls": ["https://canada.ca/.../qualify.html"],  # Same URL!
  "max_pages": 5
}
```

**Result:**
- Scrapes same 5 pages (content may have changed)
- Creates 30 chunks (same URLs generate same IDs)
- **UPSERTS** (updates) existing chunks with same IDs

**Database:** Still 30 documents (updated, not duplicated) ✅

---

## Why Upsert?

### ❌ Without Upsert (Old Behavior)
```
Ingest 1: 30 documents
Ingest 2: 30 MORE documents (duplicates!)
Ingest 3: 30 MORE documents
Total: 90 documents (60 duplicates!)
```

**Problems:**
- Wastes storage
- Pollutes search results
- Returns same answer 3 times
- Confuses similarity scoring

---

### ✅ With Upsert (New Behavior)
```
Ingest 1: 30 documents
Ingest 2: 30 documents (updated)
Ingest 3: 30 documents (updated)
Total: 30 documents (always fresh!)
```

**Benefits:**
- Keeps latest content
- No duplicates
- Clean search results
- Efficient storage

---

## When Content Changes

If the CDCP website updates their content:

```bash
# Original content:
"Income limit is $90,000"

# Website updates:
"Income limit is $95,000"

# Re-ingest same URL:
POST /rag/ingest { "base_urls": ["...qualify.html"] }

# Result: Old chunks updated with new content!
```

---

## Edge Cases

### Different Chunk Sizes

If you re-ingest with different `chunk_size`, the number of chunks may change:

```bash
# First ingest: chunk_size=512 → 30 chunks
POST /rag/ingest { "chunk_size": 512 }

# Second ingest: chunk_size=256 → 60 chunks
POST /rag/ingest { "chunk_size": 256 }
```

**Result:**
- First 30 chunks updated
- 30 new chunks added
- Total: 60 chunks

**Recommendation:** Use consistent `chunk_size` for same content.

---

### URL Changes (Redirects)

If a URL redirects to a different page:

```
Original: https://canada.ca/old-url
Redirect: https://canada.ca/new-url
```

These are treated as **different documents** (different MD5 hashes).

**Result:** Both old and new content stored.

**To clean up old content:**
```bash
DELETE /rag/clear  # Clear entire database
# OR manually delete by metadata filter
```

---

## Control Upsert Behavior

### Default: Upsert Enabled
```python
db.add_documents(..., upsert=True)  # Default
```

### Disable Upsert (Add Only)
```python
db.add_documents(..., upsert=False)
# Raises error if IDs already exist
```

---

## Best Practices

### ✅ Good: Regular Updates
```bash
# Daily cron job to refresh CDCP content
0 2 * * * curl -X POST http://localhost:8001/rag/ingest \
  -d '{"base_urls": ["..."], "max_pages": 50}'
```

### ✅ Good: Version Control
Add timestamp to metadata:
```python
metadata = {
    "title": "...",
    "url": "...",
    "ingested_at": "2025-09-30T10:00:00Z",  # Track when ingested
    "content_version": "v2"
}
```

### ⚠️ Caution: Changing Chunk Parameters
Changing `chunk_size` or `overlap` creates different chunk boundaries.

**Recommendation:**
```bash
# Clear database before changing chunk parameters
DELETE /rag/clear

# Then ingest with new parameters
POST /rag/ingest { "chunk_size": 256 }
```

---

## Monitoring Upserts

Check logs to see upsert activity:

```
INFO: Upserted 45 documents to collection
```

Check stats to see total documents:

```bash
GET /rag/stats

Response:
{
  "total_documents": 45,  # Doesn't increase if same URLs re-ingested
  "collection_name": "cdcp_documents"
}
```

---

## Summary

| Scenario | Behavior | Database Size |
|----------|----------|---------------|
| First ingest | Add new documents | Increases |
| Re-ingest same URL | Update existing documents | Stays same |
| Ingest new URL | Add new documents | Increases |
| Different chunk_size | Updates existing + adds new | May increase |

**Key Point:** Same URL = Same IDs = Upsert (Update) ✅
