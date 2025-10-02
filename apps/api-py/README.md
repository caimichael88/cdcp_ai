# CDCP AI - Python API

Python API for CDCP analysis using fine-tuned models.

## Setup

1. Install dependencies:
```bash
pip3 install -e .[ai,dev]
```

2. Run the server:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

Or use Nx:
```bash
nx serve api-py
```

## Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /cdcp/analyze` - Analyze text using fine-tuned CDCP model
- `GET /cdcp/models` - List available fine-tuned models

## Development

- Run tests: `nx test api-py`
- Lint: `nx lint api-py`
- Format: `nx format api-py`
- Type check: `nx type-check api-py`