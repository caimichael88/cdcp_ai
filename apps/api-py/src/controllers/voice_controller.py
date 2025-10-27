"""Voice controller for handling voice-to-voice conversations using LangGraph agent."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from ..agents.langgraph_agent import get_voice_agent
import logging
import base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

@router.post("/voice-call")
async def voice_conversation_agent_with_text(
    file: UploadFile= File(..., description="Audio file for voice conversation via LangGraph with text response")
):
    """
    Process voice conversation through LangGraph agent and return both text and audio:
    Audio → LangGraph Agent → [ASR → LLM → TTS] → {transcription, response_text, audio_base64}

    Returns JSON with transcription, AI response text, and base64-encoded audio.
    """
    #validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size (limit to 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    audio_content = await file.read()
    if len(audio_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 25MB")

    try:
        logger.info(f"Processing voice conversation with text via LangGraph: {file.filename}")

        voice_agent = get_voice_agent()
        result = voice_agent.process_voice_conversation(
            audio_data=audio_content,
            filename=file.filename
        )

        if result and result.get("audio_path"):
            # Read the generated audio file and encode as base64
            with open(result["audio_path"], "rb") as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                return JSONResponse({
                    "transcription": result.get("transcription", ""),
                    "response_text": result.get("response_text", ""),
                    "audio_base64": audio_base64,
                    "status": "success"
                })
        else:
            raise HTTPException(status_code=500, detail="Graph agent failed to generate response")
    except Exception as e:
        logger.error(f"Error in LangGraph voice conversation with text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def voice_health_check():
    """Health check for voice service."""
    try:
        # Get voice agent and check its status
        voice_agent = get_voice_agent()

        return {
            "status": "healthy",
            "service": "Voice (Intelligent LangGraph Agent)",
            "langgraph_status": "initialized",
            "tools": ["transcribe_audio", "process_with_llm", "synthesize_speech"]
        }
    except Exception as e:
        logger.error(f"Voice health check failed: {e}")
        return {
            "status": "error",
            "service": "Voice (Intelligent LangGraph Agent)",
            "error": str(e)
        }