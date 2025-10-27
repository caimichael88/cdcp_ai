"""CDCP analysis controller"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from ..agents.langgraph_agent import get_voice_agent
import logging
import base64

logger = logging.getLogger(__name__)

class CDCPRequest(BaseModel):
    text: str
    model: Optional[str] = "default"


class CDCPResponse(BaseModel):
    text: str
    analysis: dict
    model: str


class ModelInfo(BaseModel):
    name: str
    description: str
    version: str


class QueryRequest(BaseModel):
    query: str


router = APIRouter()


@router.post("/analyze", response_model=CDCPResponse)
async def analyze_cdcp(request: CDCPRequest):
    """Analyze text using fine-tuned CDCP model"""
    # TODO: Implement actual CDCP analysis with fine-tuned model
    return CDCPResponse(
        text=request.text,
        analysis={
            "status": "mock",
            "message": "CDCP analysis will be implemented with fine-tuned models"
        },
        model=request.model
    )


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available fine-tuned models"""
    # TODO: Return actual model information
    return [
        ModelInfo(
            name="default",
            description="Default CDCP fine-tuned model",
            version="0.1.0"
        )
    ]


@router.post("/query")
async def process_query(request: QueryRequest):
    """
    Process text query through LangGraph agent (same as voice workflow but text-only).
    This uses the intelligent agent with RAG and returns a text response with optional audio.
    """
    try:
        logger.info(f"Processing CDCP query: {request.query[:50]}...")

        # Get the voice agent (which is actually the intelligent LangGraph agent)
        agent = get_voice_agent()

        # Create a text-based message for the agent
        # Use TRANSCRIPT: prefix to skip audio transcription and go directly to RAG
        config = {"configurable": {"thread_id": "text_query"}, "recursion_limit": 25}

        # Create initial message that looks like a transcript result
        # This triggers the RAG workflow without needing audio
        from langchain_core.messages import ToolMessage
        initial_message = ToolMessage(
            content=f"TRANSCRIPT: {request.query}",
            tool_call_id="text_input"
        )

        # Invoke the graph
        result = agent.graph.invoke(
            {"messages": [initial_message]},
            config=config
        )

        logger.info(f"Graph completed with {len(result.get('messages', []))} messages")

        # Extract response text and audio path from results
        response_text = ""
        audio_path = ""

        for message in result["messages"]:
            if hasattr(message, 'content') and message.content:
                content = message.content
                if content.startswith("RAG_RESULT:"):
                    # Parse the JSON RAG result
                    try:
                        import json
                        rag_json = content.replace("RAG_RESULT:", "").strip()
                        rag_data = json.loads(rag_json)
                        response_text = rag_data.get("answer", "")
                    except:
                        response_text = content.replace("RAG_RESULT:", "").strip()
                elif content.startswith("OPENAI_RESULT:"):
                    # Use OpenAI result if available (after evaluation)
                    response_text = content.replace("OPENAI_RESULT:", "").strip()
                elif content.startswith("SEARCH_WITHOUT_RAG_RESULT:"):
                    # Use search without RAG result if evaluation failed
                    response_text = content.replace("SEARCH_WITHOUT_RAG_RESULT:", "").strip()
                elif content.startswith("AUDIO_FILE:"):
                    audio_path = content.replace("AUDIO_FILE:", "").strip()

        response_data = {
            "query": request.query,
            "answer": response_text or "No response generated",
            "status": "success"
        }

        # Include audio if generated
        if audio_path:
            try:
                with open(audio_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                    response_data["audio_base64"] = audio_base64
            except Exception as e:
                logger.warning(f"Could not read audio file: {e}")

        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"Error processing CDCP query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))