"""
Supervisor Service
Evaluates answer quality using external LLM (GPT-4, Claude, etc.)
"""

import logging
import os
from typing import Literal, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


class SupervisorService:
    """
    Service for evaluating answer quality using external LLM
    Acts as an objective judge for fine-tuned model outputs
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str = None,
        api_key: str = None
    ):
        """
        Initialize supervisor service

        Args:
            provider: LLM provider (openai or anthropic)
            model: Model name (default: gpt-4 or claude-3-sonnet)
            api_key: API key (uses env var if None)
        """
        self.provider = provider
        self._llm = None

        # Set defaults
        if model is None:
            if provider == "openai":
                self.model = "gpt-4"
            elif provider == "anthropic":
                self.model = "claude-3-sonnet-20240229"
        else:
            self.model = model

        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            if provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")

        logger.info(f"Supervisor initialized with {provider}/{self.model}")

    @property
    def llm(self):
        """Lazy load LLM"""
        if self._llm is None:
            logger.info(f"Loading supervisor LLM: {self.provider}/{self.model}")

            if self.provider == "openai":
                self._llm = ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=0.0  # Deterministic for evaluation
                )
            elif self.provider == "anthropic":
                self._llm = ChatAnthropic(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=0.0
                )

            logger.info("Supervisor LLM loaded")
        return self._llm

    def evaluate_answer(
        self,
        query: str,
        answer: str,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate if an answer is good or bad

        Args:
            query: User's original question
            answer: Generated answer to evaluate
            context: Optional RAG context used to generate answer

        Returns:
            Dict with evaluation, reason, and confidence
        """
        logger.info(f"Evaluating answer for query: {query[:50]}...")

        # Build evaluation prompt
        context_section = ""
        if context:
            context_section = f"""
Context Used:
{context[:1500]}...
"""

        eval_prompt = f"""You are an expert evaluator for a Canadian Dental Care Plan (CDCP) chatbot.

Your task is to evaluate if the generated answer is appropriate for a CDCP chatbot.

User Question: {query}
{context_section}
Generated Answer: {answer}

CRITICAL EVALUATION RULE:
The chatbot uses a FINE-TUNED MODEL that has been trained on CDCP data. The model knows CDCP information beyond what's in the context above.

DO NOT mark an answer as BAD just because it includes information not in the context.
ONLY mark BAD if the answer:
1. Is out of scope (not about CDCP/dental care)
2. CONTRADICTS the provided context
3. Is factually WRONG about CDCP

If the answer provides accurate CDCP information (even if not in context), that is GOOD.

Evaluation Criteria (in order of importance):
1. **Scope**: Is the question related to CDCP, dental care, health coverage, or dental services?
   - If NO → Answer must be BAD (out of scope for CDCP chatbot)
   - Examples of out-of-scope: weather, geography, general knowledge, other topics
2. **Query Relevance**: Does the answer directly address what the user is asking?
3. **Answer Quality**: Is the answer clear, specific, and helpful?
4. **Factual Accuracy**: Is the answer factually accurate for CDCP?
   - The model has been TRAINED on CDCP data beyond the provided context
   - **NOT in context ≠ BAD** - The model can use its training knowledge
   - Only mark BAD if the answer CONTRADICTS the context or is factually WRONG

IMPORTANT SCOPE RULES:
- This is a CDCP chatbot - ONLY evaluate answers about Canadian Dental Care Plan
- Questions about weather, geography, politics, sports, etc. are OUT OF SCOPE → Always BAD
- If the question is NOT about CDCP/dental care → the answer is BAD regardless of correctness

For IN-SCOPE CDCP questions - BE GENEROUS:
- For yes/no questions: "yes" or "no" IS GOOD (even without long explanation)
- If the answer addresses the question reasonably, mark it GOOD
- **The model has been trained on CDCP data** - it may know things not in the provided context
- **Missing from context ≠ BAD** - If the answer provides CDCP info not in context, that's OKAY
- Only mark BAD if the answer CONTRADICTS the context or is clearly wrong
- User's question may have transcription errors - interpret the intent
- A concise, direct answer is BETTER than a long, generic one
- If the answer says "Yes" or "No" and gives a brief reason, that's GOOD
- **IGNORE minor citation errors** (e.g., "document 3" when only Document 1 shown - focus on CONTENT)
- **Document reference numbers don't matter** - what matters is if the answer is factually correct

Common signs of BAD answers:
- Question is NOT about CDCP (out of scope) ← CHECK THIS FIRST
- Answer CONTRADICTS the context information (not just "different from" - actual contradiction)
- Answer is factually wrong about CDCP
- Answer doesn't address the question at all
- Answer is off-topic or irrelevant to CDCP
- Answer is vague/generic like "I don't know" when the model should know

Common signs of GOOD answers:
- Question IS about CDCP/dental care (in scope) ✓
- Answer says "yes" or "no" for yes/no questions ✓
- Answer provides relevant CDCP information ✓
- Answer uses knowledge from model training (even if not in provided context) ✓
- Clear and directly answers the question ✓
- Concise and specific to what was asked ✓
- **Content is accurate for CDCP** (even if context doesn't mention it) ✓
- Does NOT contradict the provided context ✓

Respond in this EXACT format:
EVALUATION: [GOOD or BAD]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
REASON: [One clear sentence explaining your decision]

Your evaluation:"""

        try:
            # Call supervisor LLM
            response = self.llm.invoke([SystemMessage(content=eval_prompt)])
            eval_text = response.content.strip()

            # Parse response
            evaluation = "BAD"
            confidence = "MEDIUM"
            reason = "Unknown"

            for line in eval_text.split('\n'):
                line = line.strip()
                if line.startswith("EVALUATION:"):
                    eval_value = line.split(":", 1)[1].strip().upper()
                    evaluation = "GOOD" if "GOOD" in eval_value else "BAD"
                elif line.startswith("CONFIDENCE:"):
                    conf_value = line.split(":", 1)[1].strip().upper()
                    if "HIGH" in conf_value:
                        confidence = "HIGH"
                    elif "LOW" in conf_value:
                        confidence = "LOW"
                    else:
                        confidence = "MEDIUM"
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            result = {
                "evaluation": evaluation,
                "confidence": confidence,
                "reason": reason,
                "is_good": evaluation == "GOOD",
                "query": query,
                "answer": answer
            }

            logger.info(f"Evaluation: {evaluation} ({confidence}) - {reason}")
            return result

        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            # Default to BAD on error to be safe
            return {
                "evaluation": "BAD",
                "confidence": "LOW",
                "reason": f"Evaluation error: {str(e)}",
                "is_good": False,
                "query": query,
                "answer": answer
            }


# Global singleton
_supervisor_service = None


def get_supervisor_service(
    provider: str = None,
    model: str = None
) -> SupervisorService:
    """
    Get or create supervisor service singleton

    Args:
        provider: LLM provider (openai or anthropic)
        model: Model name

    Returns:
        SupervisorService instance
    """
    global _supervisor_service

    # Use env vars if not specified
    if provider is None:
        provider = os.getenv("SUPERVISOR_PROVIDER", "openai")

    if _supervisor_service is None:
        _supervisor_service = SupervisorService(
            provider=provider,
            model=model
        )

    return _supervisor_service
