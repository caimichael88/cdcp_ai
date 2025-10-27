from __future__ import annotations
"""
TTS Service - Service layer for text-to-speech operations

Provides the TTSEngine implementation and dependency injection for controllers.
This module handles:
- Engine initialization and configuration
- Service instance management
- Fallback to fake/mock implementations for testing

The service uses the neural TTS engine by default, which combines:
- FastSpeech2 (Text → Mel spectrogram)
- HiFiGAN (Mel → Audio waveform)
- Optional loudness normalization
- Coqui TTS engine
"""
import os
import logging
from functools import lru_cache
from typing import Optional

try:
    from ..tts.ports import TTSEngine, SynthesisRequest, SynthesisResult, TTSSynthesisError
except ImportError:
    # Fallback for direct execution
    from tts.ports import TTSEngine, SynthesisRequest, SynthesisResult, TTSSynthesisError

logger = logging.getLogger(__name__)

#Service implementation
@lru_cache(maxsize=1)
def _build_coqui_tts_engine() -> TTSEngine:
    """Build Coqui TTS engine"""
    try:
        # Import Coqui TTS engine
        try:
            from ..tts.coqui_engine import CoquiTTSEngine
        except ImportError:
            # Fallback for direct execution
            from tts.coqui_engine import CoquiTTSEngine
         # Get configuration from environment
        device = os.getenv("COQUI_DEVICE", os.getenv("DEVICE", "cpu"))
        model_name = os.getenv("COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph")
        warmup_enabled = os.getenv("WARMUP_ON_STARTUP", "1") == "1"

        logger.info(f"Building Coqui TTS engine - device: {device}, model: {model_name}")

        # Build Coqui engine
        engine = CoquiTTSEngine(
            model_name=model_name,
            device=device
        )

        # Warmup (optional based on configuration)
        if warmup_enabled:
            logger.info("Warming up Coqui TTS engine...")
            engine.warmup()
            logger.info("Coqui TTS engine ready")
        else:
            logger.info("Skipping warmup (WARMUP_ON_STARTUP=0)")
        
        return engine
    
    except Exception as e:
        logger.error(f"Failed to build Coqui TTS engine: {e}")
   
@lru_cache(maxsize=1)
def _build_tts_engine() -> TTSEngine:
    """Build TTS engine based on configuration"""

    engine_type = os.getenv("TTS_ENGINE")

    # Check for Coqui TTS mode
    if engine_type == "coqui":
        logger.info("Using Coqui TTS engine")
        print("Using Coqui TTS engine")
        return _build_coqui_tts_engine()

    # Fallback to Coqui if unknown engine specified
    logger.warning(f"Unknown TTS engine '{engine_type}', falling back to Coqui")
    return _build_coqui_tts_engine()

def get_tts_service() -> TTSEngine:
    """Get TTS service instance for dependency injection"""
    return _build_tts_engine()