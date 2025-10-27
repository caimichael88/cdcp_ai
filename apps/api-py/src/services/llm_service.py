"""
LLM Service
Handles fine-tuned model inference for CDCP queries
"""

import logging
import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class LLMService:
    """Service for fine-tuned LLM inference"""

    def __init__(
        self,
        model_path: str = "./finetuned_cdcp_mps_aggressive",
        device: Optional[str] = None
    ):
        """
        Initialize LLM service with fine-tuned model

        Args:
            model_path: Path to fine-tuned model directory
            device: Device to use (cuda/mps/cpu). Auto-detected if None
        """
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"LLM Service initialized with device: {self.device}")

    @property
    def model(self):
        """Lazy load model"""
        if self._model is None:
            logger.info(f"Loading fine-tuned model from {self.model_path}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device
            )
            logger.info("Model loaded successfully")
        return self._model

    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Tokenizer loaded successfully")
        return self._tokenizer

    def generate_with_context(
        self,
        query: str,
        context: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate answer using query + RAG context

        Args:
            query: User's question
            context: Retrieved context from RAG
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated answer
        """
        logger.info(f"Generating answer with RAG context for: {query[:50]}...")

        # Create prompt with context
        prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""

        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated answer (remove prompt)
            answer = full_response.split("Answer:")[-1].strip()

            logger.info(f"Generated answer with context ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"Error generating with context: {e}")
            raise

    def generate_without_context(
        self,
        query: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate answer using only the fine-tuned model (no RAG)

        Args:
            query: User's question
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated answer
        """
        logger.info(f"Generating answer WITHOUT RAG context for: {query[:50]}...")

        # Simple prompt without context
        prompt = f"""Question: {query}

Answer:"""

        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated answer
            answer = full_response.split("Answer:")[-1].strip()

            logger.info(f"Generated answer without context ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"Error generating without context: {e}")
            raise

    def unload_model(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Model unloaded from memory")


# Global singleton
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
