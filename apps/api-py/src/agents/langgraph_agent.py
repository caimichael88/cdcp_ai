"""Intelligent Langgraph Agent for the CDCP App"""

from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import tempfile
import logging
import json
import os
import base64


from ..services.asr_service import asr_service
from ..services.tts_service import get_tts_service
from ..tts.ports import SynthesisRequest
from ..services.rag_service import get_rag_service
from ..services.llm_service import get_llm_service


logger = logging.getLogger(__name__)


@tool
def transcribe_audio(audio_data: str, file_name: str = "audio.wav") -> str:
    """Transcribe audio data to text using ASR service."""
    import asyncio

    try:
        logger.info(f"Tool: Transcribing audio file: {file_name}")

        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)

        # Run async function in sync context
        async def _transcribe():
            return await asr_service.transcribe_audio(
                audio_file=audio_bytes,
                filename=file_name,
                model_name=os.getenv("ASR_MODEL")
            )

        result = asyncio.run(_transcribe())
        return f"TRANSCRIPT: {result.get('text', '')}"
    except Exception as e:
        error_msg = f"Error transcribing audio: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@tool
def synthesize_speech(text: str) -> str:
    """Convert text to speech using TTS service and return audio file path."""
    try:
        logger.info(f"Tool: Synthesizing speech for: {text[:50]}...")

        # Use existing TTS service
        tts_engine = get_tts_service()

        # Create synthesis request
        request = SynthesisRequest(
            text=text,
            voice_id="en_female_1",
            sample_rate=22050
        )

        # Synthesize audio
        result = tts_engine.synthesize_sync(request)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with open(temp_file.name, "wb") as f:
            f.write(result.audio)

        audio_path = temp_file.name
        logger.info(f"Tool: Audio synthesized to: {audio_path}")
        return f"AUDIO_FILE: {audio_path}"

    except Exception as e:
        error_msg = f"Error synthesizing speech: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@tool
def rag_search_and_generate(query: str) -> str:
    """
    Tool 1: RAG vector search + Fine-tuned model answer generation.
    Searches vector DB and generates answer using fine-tuned model with RAG context.
    Returns JSON string with answer, context, and sources.
    """

    logger.info(f"[RAG Tool] Query: {query[:50]}...")

    try:
        # Get singleton services
        rag_service = get_rag_service()
        llm_service = get_llm_service()

        # Vector Search
        logger.info("[RAG Tool] Searching vector database...")
        search_results = rag_service.search(query.strip(), n_results=3)

        # Check if we have results
        if not search_results.results or len(search_results.results) == 0:
            logger.warning("[RAG Tool] No RAG results found")
            return json.dumps({"status": "no_results"})

        # Build context from top results
        context_parts = []
        sources = []
        for i, doc in enumerate(search_results.results[:3], 1):
            context_parts.append(
                f"[Document {i}] (Relevance: {doc.similarity_score:.1%})\n"
                f"Title: {doc.title}\n"
                f"Content: {doc.content}"
            )
            sources.append(doc.url)

        context = "\n\n".join(context_parts)
        logger.info(f"[RAG Tool] Found {len(search_results.results)} documents")

        # Generate answer with Fine-tuned Model + Context
        logger.info("[RAG Tool] Generating answer with fine-tuned model...")
        answer = llm_service.generate_with_context(
            query=query,
            context=context,
            max_length=300,
            temperature=0.7
        )

        logger.info(f"[RAG Tool] Answer generated ({len(answer)} chars)")

        result = json.dumps({
            "query": query,
            "answer": answer,
            "context": context,
        })
        return f"RAG_RESULT: {result}"
    
    except Exception as e:
        error_msg = f"Error Rag search result: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


@tool
def evaluate_answer(query: str, answer: str, context: str) -> str:
    """
    Tool 2: Evaluate answer quality using OpenAI supervisor.
    Returns JSON string with evaluation result (GOOD/BAD) and reason.
    """
    from ..services.supervisor_service import get_supervisor_service

    logger.info("[Evaluate Tool] Evaluating answer quality...")

    try:
        supervisor = get_supervisor_service()

        evaluation = supervisor.evaluate_answer(
            query=query,
            answer=answer,
            context=context
        )

        logger.info(f"[Evaluate Tool] {evaluation['evaluation']} - {evaluation['reason']}")


        result = json.dumps({
            "evaluation": evaluation["evaluation"],
            "reason": evaluation["reason"],
            "is_good": evaluation["is_good"],
            "query": evaluation["query"],
            "answer": evaluation["answer"]
        })

        return f"EVALUATE_RESULT: {result}"

    except Exception as e:
        error_msg = f"Error Evaluation: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


@tool
def generate_without_rag(query: str) -> str:
    """
    Tool 3: Generate answer using fine-tuned model without RAG context.
    Fallback when RAG results are poor or evaluation fails.
    Returns JSON string with answer.
    """


    logger.info("[Fallback Tool] Generating without RAG...")

    try:
        llm_service = get_llm_service()

        answer = llm_service.generate_without_context(
            query=query,
            max_length=300,
            temperature=0.7
        )

        logger.info(f"[Fallback Tool] Generated ({len(answer)} chars)")

        result = json.dumps({
            "status": "success",
            "answer": answer
        })

        return f"SEARCH_WITHOUT_RAG_RESULT: {result}"

    except Exception as e:
        error_msg = f"Error search model directly without rag: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


@tool
def generate_with_openai(query: str) -> str:
    """
    Tool 4: Generate answer using OpenAI when RAG answer is poor.
    Fallback to external LLM for better quality answers.
    Returns JSON string with answer.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    logger.info("[OpenAI Fallback Tool] Generating answer with OpenAI...")

    try:
        # Initialize OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        llm = ChatOpenAI(
            model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper
            api_key=api_key,
            temperature=0.7
        )

        # Create messages
        system_prompt = """You are a helpful assistant answering questions about the Canadian Dental Care Plan (CDCP).
Provide clear, accurate, and concise answers based on your knowledge.
Keep answers between 2-4 sentences unless more detail is requested."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        # Generate answer
        response = llm.invoke(messages)
        answer = response.content

        logger.info(f"[OpenAI Fallback Tool] Generated ({len(answer)} chars)")

        result = json.dumps({
            "status": "success",
            "answer": answer,
            "source": "openai"
        })

        return f"OPENAI_RESULT: {result}"

    except Exception as e:
        error_msg = f"Error generating with OpenAI: {e}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


class IntelligentAIAgent:
    def __init__(self):
        self.tools = [transcribe_audio, synthesize_speech, rag_search_and_generate, evaluate_answer, generate_without_rag, generate_with_openai]
        self.tool_node = ToolNode(self.tools)
        self.memory = MemorySaver()
        self.graph = self._create_graph()
        print("Intelligent AI Agent initialized")
    
    def _create_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("coordinator", self._coordinate_tools)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "coordinator")

        return workflow.compile(checkpointer=self.memory)
    
    def _coordinate_tools(self, state: MessagesState):
        """Intelligent coordinator that decide which tools to call"""
        messages = state["messages"]
        last_message = messages[-1]

        print(f"\n=== COORDINATOR CALLED ===")
        print(f"Coordinator: Processing message type: {type(last_message).__name__}")
        print(f"Coordinator: Total messages in state: {len(messages)}")

        if isinstance(last_message, HumanMessage):
            content = last_message.content
            if "AUDIO_DATA:" in content:
                #Extract audio data and filename from content
                parts = content.split("AUDIO_DATA:")
                
                if len(parts)>1:
                    audio_part = parts[1]
                    if "|" in audio_part:
                        audio_data, filename = audio_part.split("|", 1)
                    else:
                        audio_data, filename = audio_part, "audio.wav"

                    print(f"Coordinator: Found audio data, starting transcription for {filename}")
                    tool_call = {
                        "name": "transcribe_audio",
                        "args": {"audio_data": audio_data, "file_name": filename},
                        "id": "transcribe_1"
                    }
                    print(f"Coordinator: Returning AIMessage with tool_calls")
                    return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

        # Check for tool results and decide next step
        # Look for the most recent tool result or ToolMessage
        tool_result_msg = None
        print(f"Coordinator: Checking {len(messages)} messages for tool results")
        for i, msg in enumerate(reversed(messages)):
            print(f"  Message {i}: type={type(msg).__name__}, has_content={hasattr(msg, 'content')}, content_preview={msg.content[:50] if hasattr(msg, 'content') and msg.content else 'None'}")
            if hasattr(msg, 'content') and msg.content and any(msg.content.startswith(prefix) for prefix in ["TRANSCRIPT:", "AUDIO_FILE:", "ERROR:", "RAG_RESULT:", "EVALUATE_RESULT:", "SEARCH_WITHOUT_RAG_RESULT:", "OPENAI_RESULT:"]):
                tool_result_msg = msg
                print(f"  ✓ Found tool result message!")
                break
        
        if tool_result_msg:
            content = tool_result_msg.content
            logger.info(f"Coordinator: Found tool result: {content[:50]}...")
            print(f"DEBUG: Full content: {content[:200]}...")  # Debug print

            # If we just got a transcription, use RAG search for all queries
            if content.startswith("TRANSCRIPT:"):
                transcript = content.replace("TRANSCRIPT: ", "")
                tool_call = {
                    "name": "rag_search_and_generate",
                    "args": {"query": transcript},
                    "id": "rag_1"
                }
                logger.info("Coordinator: Starting rag_search_and_generate")
                return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
            
            elif content.startswith("AUDIO_FILE:"):
                logger.info("Coordinator: Process complete - audio file generated, letting should_continue end the workflow")
                # Return the current state unchanged - no new messages
                return {"messages": messages}
            
            elif content.startswith("ERROR:"):
                logger.error(f"Coordinator: Tool error encountered: {content}")
                return {"messages": [AIMessage(content="Error in processing")]}
            
            elif content.startswith("RAG_RESULT:"):
                result = content.replace("RAG_RESULT:", "", 1).strip()
                parsed_result = json.loads(result)
                print(f"Coordinator: Got RAG results, sending to evaluate_answer")

                # Evaluate the answer quality
                tool_call = {
                    "name": "evaluate_answer",
                    "args": {"query": parsed_result["query"], "answer": parsed_result["answer"], "context": parsed_result["context"]},
                    "id": "evaluate_rag"
                }
                return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
            
            elif content.startswith("EVALUATE_RESULT:"):
                result = content.replace("EVALUATE_RESULT:", "", 1).strip()
                parsed_result = json.loads(result)
                print(f"Coordinator: Got Evaluate results: {result}")
                #Evaluation result is good, go to last step and get the voice file
                print(f"is_good: {parsed_result['is_good']} (type: {type(parsed_result['is_good'])})")
                print(f"evaluation: {parsed_result['evaluation']}")
                if parsed_result["is_good"]:
                    tool_call = {
                        "name": "synthesize_speech",
                        "args": {"text": parsed_result["answer"]},
                        "id": "tts_rag"
                    }
                    return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
                else:
                #Evaluation result is bad, use OpenAI as fallback for better answer
                    print(f"Coordinator: RAG answer was BAD, using OpenAI fallback")
                    tool_call = {
                        "name": "generate_with_openai",
                        "args": {"query": parsed_result["query"]},
                        "id": "openai_fallback"
                    }
                    return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
            elif content.startswith("OPENAI_RESULT:"):
                result = content.replace("OPENAI_RESULT:", "", 1).strip()
                parsed_result = json.loads(result)
                print(f"Coordinator: Got OpenAI answer, sending to TTS")
                tool_call = {
                    "name": "synthesize_speech",
                    "args": {"text": parsed_result["answer"]},
                    "id": "tts_openai"
                }
                return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

            elif content.startswith("SEARCH_WITHOUT_RAG_RESULT:"):
                result = content.replace("SEARCH_WITHOUT_RAG_RESULT:", "", 1).strip()
                parsed_result = json.loads(result)
                tool_call = {
                    "name": "synthesize_speech",
                    "args": {"text": parsed_result["answer"]},
                    "id": "tts_rag"
                }
                return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

        #No more tools needed
        logger.info("Coordinator: No action needed, ending")
        return {"messages": [AIMessage(content="No processing needed")]}

    def _should_continue(self, state: MessagesState):
        """Decide whether to conintue with more to end"""
        messages = state["messages"]
        last_message = messages[-1]
        print(f"Should continue: Last message type: {type(last_message).__name__}")
        print(f"Should continue: Has tool_calls attr: {hasattr(last_message, 'tool_calls')}")
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Log tool calls without the full audio_data
            tool_calls_summary = []
            for tc in last_message.tool_calls:
                if isinstance(tc, dict):
                    args_summary = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in tc.get('args', {}).items()}
                    tool_calls_summary.append({'name': tc.get('name'), 'args': args_summary, 'id': tc.get('id')})
                else:
                    tool_calls_summary.append(str(tc)[:100])
            print(f"Should continue: tool_calls: {tool_calls_summary}")
        if hasattr(last_message, "content"):
            print(f"Should continue: Content preview: {(last_message.content[:50] + '...') if last_message.content and len(last_message.content) > 50 else last_message.content}")

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("Should continue: YES - Tool calls pending execution")
            return "continue"
        
        # Check for completion messages - these should end
        if hasattr(last_message, 'content') and last_message.content:
            content = last_message.content
            if content in ["Voice processing complete", "No processing needed", "Error in processing"]:
                print(f"Should continue: NO - Process ended with: {content}")
                return "end"

        # Check if we have an audio file in any message - this means completion
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content and msg.content.startswith("AUDIO_FILE:"):
                print("Should continue: NO - AUDIO_FILE found, process complete")
                return "end"

         # If last message is a tool result that needs further processing, continue
        if hasattr(last_message, 'content') and last_message.content:
            content = last_message.content
            if content.startswith(("TRANSCRIPT:", "RAG_RESULT:", "EVALUATE_RESULT:", "SEARCH_WITHOUT_RAG_RESULT:", "OPENAI_RESULT:")):
                print(f"Should continue: YES - Tool result needs processing: {content.split(':')[0]}")
                return "continue"
            elif content.startswith("ERROR:"):
                print("Should continue: NO - Error occurred")
                return "end"

        # Default to end to prevent infinite loops
        print("Should continue: NO - Default to end (safety)")
        return "end"
    
    def process_voice_conversation(self, audio_data: bytes, filename = "audio.wav", thread_id: str = "default") -> str:
        """Process a voice conversation through the intelligent agent."""
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(f"Processing voice conversation: {filename}")

        # Encode audio data as base64 to include in message content
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Create initial message with encoded audio data in content
        initial_message = HumanMessage(
            content=f"AUDIO_DATA:{audio_b64}|{filename}"
        )

        # Add recursion limit to config - increase for debugging
        config.update({"recursion_limit": 25})

        result = self.graph.invoke(
            {"messages": [initial_message]},
            config=config
        )

        logger.info(f"Graph completed with {len(result.get('messages', []))} messages")

        # Extract transcription, response text, and audio path from the results
        transcription = ""
        response_text = ""
        audio_path = ""

        for message in result["messages"]:
            if hasattr(message, 'content') and message.content:
                content = message.content
                if content.startswith("TRANSCRIPT:"):
                    transcription = content.replace("TRANSCRIPT:", "").strip()
                elif content.startswith("RAG_RESULT:"):
                    # Parse the JSON to extract just the answer
                    try:
                        rag_json = content.replace("RAG_RESULT:", "").strip()
                        rag_data = json.loads(rag_json)
                        response_text = rag_data.get("answer", "")
                    except Exception as e:
                        logger.warning(f"Failed to parse RAG_RESULT JSON: {e}")
                        response_text = content.replace("RAG_RESULT:", "").strip()
                elif content.startswith("OPENAI_RESULT:"):
                    # OpenAI result takes precedence (it's the evaluated/improved answer)
                    try:
                        openai_json = content.replace("OPENAI_RESULT:", "").strip()
                        openai_data = json.loads(openai_json)
                        response_text = openai_data.get("answer", "")
                    except Exception as e:
                        logger.warning(f"Failed to parse OPENAI_RESULT JSON: {e}")
                        response_text = content.replace("OPENAI_RESULT:", "").strip()
                elif content.startswith("SEARCH_WITHOUT_RAG_RESULT:"):
                    # Fallback if evaluation fails
                    if not response_text:  # Only use if we don't have a better answer
                        try:
                            search_json = content.replace("SEARCH_WITHOUT_RAG_RESULT:", "").strip()
                            search_data = json.loads(search_json)
                            response_text = search_data.get("answer", "")
                        except Exception as e:
                            logger.warning(f"Failed to parse SEARCH_WITHOUT_RAG_RESULT JSON: {e}")
                            response_text = content.replace("SEARCH_WITHOUT_RAG_RESULT:", "").strip()
                elif content.startswith("AUDIO_FILE:"):
                    audio_path = content.replace("AUDIO_FILE:", "").strip()

        print(f"Extracted - Transcription: {transcription[:50] if transcription else 'None'}..., Response: {response_text[:50] if response_text else 'None'}..., Audio: {audio_path}")
        if not audio_path:
            logger.error("No audio file was generated by the workflow")
            return None

        return {
            "transcription": transcription,
            "response_text": response_text,
            "audio_path": audio_path
        }
    
# Global instance
voice_agent = None

def get_voice_agent() -> IntelligentAIAgent:
    """Get or create the global voice agent instance."""
    global voice_agent
    if voice_agent is None:
        voice_agent = IntelligentAIAgent()
    
    return voice_agent