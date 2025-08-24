import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import HTTPException

from clients.base import get_llm_client_and_model_details
from config.models import ChatRequest, Message
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SmartRouter:
    def __init__(self):
        pass

    def _has_vision_content(self, messages: List[Message]) -> bool:
        """Check if any message in the conversation contains images"""
        for message in messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if part.get("type") == "image_url":
                        logger.info("🖼️ Vision content detected")
                        return True
        return False


    def _should_use_searching_model(self, messages: List[Message]) -> bool:
        """Check if the user message contains keywords that trigger searching model routing"""
        keywords = settings.search_keywords or ["research", "look up", "lookup", "look-up", "search", "verify", "`"]
        
        # Check the last user message for keywords
        for message in reversed(messages):
            if message.role == "user":
                content_text = self._get_text_from_content(message.content).lower()
                for keyword in keywords:
                    if keyword in content_text:
                        logger.info(f"🔍 Keyword '{keyword}' detected - routing to searching_model")
                        return True
                break
        return False

    def _extract_system_prompt_from_messages(
        self, messages: List[Message]
    ) -> tuple[List[Message], Optional[str]]:
        """Extract system prompt from messages if present, return cleaned messages and system prompt"""
        system_prompt_override: Optional[str] = None
        cleaned_messages = []

        for message in messages:
            if message.role == "system":
                # Found a system message - use it as override
                if isinstance(message.content, str):
                    system_prompt_override = message.content
                    logger.info(
                        f"📝 System prompt override detected: '{system_prompt_override[:100]}...'"
                    )
                # Don't add system messages to cleaned_messages - they're handled separately
            else:
                cleaned_messages.append(message)

        return cleaned_messages, system_prompt_override



    async def route_and_process(
        self, chat_request: ChatRequest, auth_header: str
    ) -> Dict[str, Any]:
        """Handle non-streaming requests"""
        messages = chat_request.messages
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        # Extract system prompt override from messages
        cleaned_messages, system_prompt_override = self._extract_system_prompt_from_messages(
            messages
        )

        if not cleaned_messages:
            raise HTTPException(status_code=400, detail="No user messages provided.")

        last_message = cleaned_messages[-1]
        user_prompt_text = self._get_text_from_content(last_message.content)

        logger.info(f"🚀 Processing request: '{user_prompt_text[:100]}...'")
        if system_prompt_override:
            logger.info("📝 Using custom system prompt from OpenWebUI")

        # Simple routing: search keywords = searching_model, otherwise = working_model
        if self._should_use_searching_model(cleaned_messages):
            logger.info("🔍 Routing to searching_model for research query")
            target_model = "searching_model"
        else:
            target_model = "working_model"

        # Ensure we have vision capability if needed
        has_vision = self._has_vision_content(cleaned_messages)
        if has_vision:
            logger.info("🖼️ Vision content detected - using vision-capable model")

        logger.info(f"🎯 Selected model: {target_model}")

        # Get client and model details
        client, model_id, model_type, api_params, default_system_prompt = (
            get_llm_client_and_model_details(target_model)
        )

        if not client or not model_id:
            logger.warning(f"⚠️ Target model '{target_model}' not available, trying fallback.")
            client, model_id, model_type, api_params, default_system_prompt = (
                get_llm_client_and_model_details("working_model")
            )
            if not client or not model_id:
                raise HTTPException(status_code=500, detail="No valid LLM client available")
            target_model = "working_model"

        provider_name = getattr(client, "provider_name", "unknown")
        logger.info(f"🔧 Using: {target_model} ({provider_name}: {model_id})")

        # Choose which system prompt to use: only use OpenWebUI prompt, no fallback
        final_system_prompt = system_prompt_override

        if final_system_prompt:
            logger.debug(f"📝 Using OpenWebUI system prompt: '{final_system_prompt[:100]}...'")
        else:
            logger.debug("📝 No system prompt provided - using default behavior")

        try:
            temperature = chat_request.temperature if chat_request.temperature is not None else 0.7
            message_dicts = [msg.dict() for msg in cleaned_messages]

            response = await client.generate_response(
                model_id=model_id,
                messages=message_dicts,
                temperature=temperature,
                api_parameters=api_params or {},
                system_prompt=final_system_prompt,  # Use only OpenWebUI prompt, could be None
            )

            logger.info("✅ Response generated successfully")
            return self._format_openai_response(response, target_model)

        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

    async def route_and_process_streaming(
        self, chat_request: ChatRequest, auth_header: str
    ) -> AsyncGenerator[str, None]:
        """Handle streaming requests"""
        messages = chat_request.messages
        if not messages:
            yield f"data: {json.dumps({'error': 'No messages provided'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Extract system prompt override from messages
        cleaned_messages, system_prompt_override = self._extract_system_prompt_from_messages(
            messages
        )

        if not cleaned_messages:
            yield f"data: {json.dumps({'error': 'No user messages provided'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        last_message = cleaned_messages[-1]
        user_prompt_text = self._get_text_from_content(last_message.content)

        logger.info(f"🚀 Processing streaming request: '{user_prompt_text[:100]}...'")
        if system_prompt_override:
            logger.info("📝 Using custom system prompt from OpenWebUI")

        # Simple routing: search keywords = searching_model, otherwise = working_model
        if self._should_use_searching_model(cleaned_messages):
            logger.info("🔍 Routing to searching_model for research query")
            target_model = "searching_model"
        else:
            target_model = "working_model"

        # Ensure we have vision capability if needed
        has_vision = self._has_vision_content(cleaned_messages)
        if has_vision:
            logger.info("🖼️ Vision content detected - using vision-capable model")

        logger.info(f"🎯 Selected model for streaming: {target_model}")

        # Get client and model details
        client, model_id, model_type, api_params, default_system_prompt = (
            get_llm_client_and_model_details(target_model)
        )

        if not client or not model_id:
            logger.warning(f"⚠️ Target model '{target_model}' not available, trying fallback.")
            client, model_id, model_type, api_params, default_system_prompt = (
                get_llm_client_and_model_details("working_model")
            )
            if not client or not model_id:
                yield f"data: {json.dumps({'error': 'No valid LLM client available'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            target_model = "working_model"

        provider_name = getattr(client, "provider_name", "unknown")
        logger.info(f"🔧 Using for streaming: {target_model} ({provider_name}: {model_id})")

        # Choose which system prompt to use: only use OpenWebUI prompt, no fallback
        final_system_prompt = system_prompt_override

        if final_system_prompt:
            logger.debug(f"📝 Using OpenWebUI system prompt: '{final_system_prompt[:100]}...'")
        else:
            logger.debug("📝 No system prompt provided - using default behavior")

        try:
            temperature = chat_request.temperature if chat_request.temperature is not None else 0.7
            message_dicts = [msg.dict() for msg in cleaned_messages]

            # Check if client supports streaming
            if hasattr(client, "supports_streaming") and client.supports_streaming():
                logger.info(f"✅ Using native streaming for {target_model}")

                async for chunk in client.generate_streaming_response(
                    model_id=model_id,
                    messages=message_dicts,
                    temperature=temperature,
                    api_parameters=api_params or {},
                    system_prompt=final_system_prompt,
                ):
                    # Format the chunk for OpenAI compatibility
                    formatted_chunk = self._format_streaming_chunk(chunk, target_model)
                    yield f"data: {json.dumps(formatted_chunk)}\n\n"

                    # Check if this is the final chunk
                    if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
                        break

            else:
                logger.info(f"⚠️ Fallback to simulated streaming for {target_model}")

                # Fallback: generate full response and simulate streaming
                response = await client.generate_response(
                    model_id=model_id,
                    messages=message_dicts,
                    temperature=temperature,
                    api_parameters=api_params or {},
                    system_prompt=final_system_prompt,
                )

                # Simulate streaming by chunking the response
                full_content = ""
                if isinstance(response, dict) and "choices" in response:
                    full_content = response["choices"][0]["message"]["content"]

                # Stream in chunks of ~50 characters
                chunk_size = 50
                for i in range(0, len(full_content), chunk_size):
                    chunk_content = full_content[i : i + chunk_size]

                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": target_model,
                        "choices": [
                            {"index": 0, "delta": {"content": chunk_content}, "finish_reason": None}
                        ],
                    }

                    yield f"data: {json.dumps(chunk)}\n\n"

                    # Small delay to simulate real streaming
                    await asyncio.sleep(0.05)

                # Send final chunk
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"

            # Send completion marker
            yield "data: [DONE]\n\n"
            logger.info("✅ Streaming response completed successfully")

        except Exception as e:
            logger.error(f"❌ Error generating streaming response: {e}")
            error_chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": target_model,
                "choices": [
                    {"index": 0, "delta": {"content": f"Error: {str(e)}"}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    def _format_streaming_chunk(self, chunk: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Format streaming chunk to ensure OpenAI compatibility"""

        # Add debug prefix only in debug mode
        if settings.log_level.upper() == "DEBUG":
            choices = chunk.get("choices", [{}])
            if choices and "delta" in choices[0] and "content" in choices[0]["delta"]:
                content = choices[0]["delta"]["content"]
                if content and not content.startswith(f"{model_name} - "):
                    # Only add prefix to the first chunk with content
                    choices[0]["delta"]["content"] = f"{model_name} - {content}"

        return chunk

    def _get_text_from_content(self, content: Any) -> str:
        """Extract text from message content"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        return ""

    def _format_openai_response(self, llm_response: Any, model_name: str) -> Dict[str, Any]:
        """Format response to OpenAI-compatible format"""
        # Handle image generation responses
        if (
            isinstance(llm_response, dict)
            and "data" in llm_response
            and llm_response.get("object") == "list"
        ):
            return {
                "id": llm_response.get("id", f"img-{uuid.uuid4()}"),
                "object": "chat.completion",
                "created": llm_response.get("created", int(time.time())),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I've generated an image based on your request.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10},
                "images": llm_response.get("data", []),
            }

        # Handle error responses
        if isinstance(llm_response, dict) and "error" in llm_response:
            return llm_response

        # Standard chat completion response
        content = "An error occurred or no content was generated."
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Extract content from response
        if isinstance(llm_response, dict) and llm_response.get("choices"):
            content = llm_response["choices"][0]["message"]["content"]
            if llm_response.get("usage"):
                usage = {
                    "prompt_tokens": llm_response["usage"].get("prompt_tokens", 0),
                    "completion_tokens": llm_response["usage"].get("completion_tokens", 0),
                    "total_tokens": llm_response["usage"].get("total_tokens", 0),
                }
        elif isinstance(llm_response, dict):
            if "choices" in llm_response:
                content = llm_response["choices"][0]["message"]["content"]
                usage = llm_response.get("usage", usage)
            elif "message" in llm_response:
                content = llm_response["message"]["content"]

        # Add debug prefix only in debug mode
        if settings.log_level.upper() == "DEBUG" and not content.startswith(f"{model_name} - "):
            content = f"{model_name} - {content}"

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }


router = SmartRouter()
