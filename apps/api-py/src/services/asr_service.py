"""Automatic Speech Recognition service using OpenAI Whisper."""
import os
import tempfile
from typing import Optional, Dict, Any
import whisper
import torch
from pathlib import Path

class ASRService:
    """service for automatic speech recoginition using Whisper"""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self.available_models = ["tiny", "base", "small", "medium", "large"]
        self.default_model = "base"  # Changed from "medium" to "base" for faster downloads (74MB vs 769MB)
    
    def _get_model(self, model_name: str = "base"):
        """load and cache Whisper model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. choose from {self.available_models}")
        
        if model_name not in self._models:
            print(f"Loading whipser model: {model_name}")
            self._models[model_name] = whisper.load_model(model_name)
        
        return self._models[model_name]

    async def transcribe_audio(
            self,
            audio_file: bytes,
            filename: str,
            model_name: str = "base",
            language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_file: Audio file content as bytes
            filename: Original filename for format detection
            model_name: Whisper model to use
            language: Optional language code (e.g., 'en', 'es', 'fr')

        Returns:
            Dictionary with transcription results
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                tmp_file.write(audio_file)
                tmp_file_path = tmp_file.name
            
            try:
                #load model
                model = self._get_model(model_name)
                
                options = {}
                if language: 
                    options["language"] = language
                
                result = model.transcribe(tmp_file_path, **options)

                return {
                    "text": result["text"].strip(),
                    "language": result.get("language", "unknown"),
                    "segments": [
                        {
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"].strip()
                        }
                        for segment in result.get("segments", [])
                    ],
                    "model_used": model_name,
                    "file_name": filename
                }
            except Exception as e:
                raise Exception(f"Unable to return segments: {e}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        except Exception as e:
            raise Exception(f"Transcription failed: {e}")
    
    def preload_default_model(self) -> None:
        """preload the default model at startup for faster first request"""
        try:
            print(f"preloading Whisper model: {self.default_model}")
            self._get_model(self.default_model)
            print(f"Whisper model '{self.default_model}' preloaded successfully")
        except Exception as e:
            print(f"Failed to preload Whisper model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """get information about available model"""
        return {
            "available_models": self.available_models,
            "default_model": self.default_model,
            "loaded_models": list(self._models.keys()),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_descriptions": {
                "tiny": "39 MB, ~32x realtime speed",
                "base": "74 MB, ~16x realtime speed",
                "small": "244 MB, ~6x realtime speed",
                "medium": "769 MB, ~2x realtime speed",
                "large": "1550 MB, ~1x realtime speed"
            }
        }       

#Global service instance
asr_service = ASRService()