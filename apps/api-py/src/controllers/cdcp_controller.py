"""CDCP analysis controller"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional


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