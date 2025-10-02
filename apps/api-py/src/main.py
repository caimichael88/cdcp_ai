"""
CDCP AI - Python API
Main FastAPI application entry point for fine-tuned CDCP models
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .controllers import cdcp_controller, health_controller, rag_controller

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting CDCP AI API...")
    print("📦 Loading fine-tuned models...")
    # TODO: Preload your fine-tuned models here
    yield
    # Shutdown
    print("👋 Shutting down CDCP AI API...")


# Create FastAPI application
app = FastAPI(
    title="CDCP AI API",
    description="Python API for CDCP analysis using fine-tuned models",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:4200"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_controller.router, prefix="/health", tags=["health"])
app.include_router(cdcp_controller.router, prefix="/cdcp", tags=["cdcp"])
app.include_router(rag_controller.router, prefix="/rag", tags=["rag"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "CDCP AI API",
        "version": "0.1.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
