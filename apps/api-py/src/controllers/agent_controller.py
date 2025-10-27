"""
Agent Controller
API endpoints for intelligent CDCP agent with supervised RAG
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


class QueryRequest(BaseModel):
    """Request model for supervised query"""
    query: str = Field(..., description="User's question")


class QueryResponse(BaseModel):
    """Response model for supervised query"""
    query: str
    answer: str
    source: str
    metadata: dict


@router.post("/query", response_model=QueryResponse)
async def supervised_query(request: QueryRequest):
    """
    Process query through supervised RAG pipeline:
    1. Vector search for relevant documents
    2. Fine-tuned model generates answer with RAG context
    3. OpenAI supervisor evaluates answer quality
    4. Falls back to fine-tuned model without RAG if needed
    """
    try:
        from ..agents.langgraph_agent import search_document_with_supervision

        logger.info(f"Processing supervised query: {request.query[:50]}...")

        # Call the supervised RAG tool
        result = search_document_with_supervision(request.query)

        # Parse the result
        if result.startswith("ERROR:"):
            raise HTTPException(status_code=500, detail=result)

        # Extract answer and source
        lines = result.split("\n\n")
        answer = ""
        source = "unknown"
        metadata = {}

        for line in lines:
            if line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("[Source:"):
                source = line.replace("[Source:", "").replace("]", "").strip()

        if not answer:
            answer = result  # Fallback to full result

        return QueryResponse(
            query=request.query,
            answer=answer,
            source=source,
            metadata={
                "pipeline": "supervised_rag",
                "evaluator": "openai"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in supervised query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
