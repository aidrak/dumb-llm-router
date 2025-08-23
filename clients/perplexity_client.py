import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from clients.base import BaseLLMClient
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PerplexityClient(BaseLLMClient):
    def __init__(self):
        super().__init__("perplexity")
        self.base_url = "https://api.perplexity.ai"

    def initialize_client(self) -> bool:
        if not settings.perplexity_api_key:
            logger.warning("PERPLEXITY_API_KEY not set. Perplexity models will not be available.")
            return False

        try:
            logger.info("✅ Perplexity client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Perplexity client: {e}")
            return False

    def supports_streaming(self) -> bool:
        """Perplexity supports streaming"""
        return True

    async def generate_response(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        api_parameters: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Build messages for Perplexity API
        llm_messages = []
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            # Perplexity API expects text content only
            if isinstance(msg["content"], list):
                # Extract text from multimodal content
                text_parts = []
                for part in msg["content"]:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                content = " ".join(text_parts)
            else:
                content = msg["content"]

            llm_messages.append({"role": msg["role"], "content": content})

        # Prepare basic request payload - start minimal and add parameters that exist
        payload = {
            "model": model_id,
            "messages": llm_messages,
        }

        # Add temperature if not excluded
        if not api_parameters.get("exclude_temperature", False):
            payload["temperature"] = temperature

        # Add supported parameters based on research
        if "max_tokens" in api_parameters:
            payload["max_tokens"] = api_parameters["max_tokens"]

        if "top_p" in api_parameters:
            payload["top_p"] = api_parameters["top_p"]

        if "presence_penalty" in api_parameters:
            payload["presence_penalty"] = api_parameters["presence_penalty"]

        if "stream" in api_parameters:
            payload["stream"] = api_parameters["stream"]

        # Search-related parameters (research shows these are supported)
        if "search_domain_filter" in api_parameters:
            payload["search_domain_filter"] = api_parameters["search_domain_filter"]

        if "search_recency_filter" in api_parameters:
            payload["search_recency_filter"] = api_parameters["search_recency_filter"]

        # Note: return_citations is now automatic and always enabled
        # search_context_size might be supported but let's test basic first

        headers = {
            "Authorization": f"Bearer {settings.perplexity_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient() as client:
                # Log the request for debugging
                logger.debug("🔍 Perplexity API Request:")
                logger.debug(f"   URL: {self.base_url}/chat/completions")
                logger.debug(f"   Model: {model_id}")
                logger.debug(f"   Messages: {len(llm_messages)}")
                logger.debug(f"   Payload keys: {list(payload.keys())}")

                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=60.0
                )

                # Log response details for debugging
                logger.debug(f"📤 Perplexity Response Status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"❌ Perplexity Error Response: {response.text}")

                response.raise_for_status()

                response_data = response.json()
                logger.info(f"✅ Received response from Perplexity {model_id}")

                # Convert Perplexity response format to OpenAI-compatible format
                return self._convert_perplexity_response(response_data, model_id)

        except Exception as e:
            logger.error(f"❌ Error calling Perplexity {model_id}: {e}")
            logger.error(f"❌ Request payload was: {payload}")
            raise

    async def generate_streaming_response(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        api_parameters: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # Build messages for Perplexity API
        llm_messages = []
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            # Perplexity API expects text content only
            if isinstance(msg["content"], list):
                # Extract text from multimodal content
                text_parts = []
                for part in msg["content"]:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                content = " ".join(text_parts)
            else:
                content = msg["content"]

            llm_messages.append({"role": msg["role"], "content": content})

        # Prepare streaming request payload
        payload = {
            "model": model_id,
            "messages": llm_messages,
            "stream": True,  # Enable streaming
        }

        # Add temperature if not excluded
        if not api_parameters.get("exclude_temperature", False):
            payload["temperature"] = temperature

        # Add supported parameters
        if "max_tokens" in api_parameters:
            payload["max_tokens"] = api_parameters["max_tokens"]

        if "top_p" in api_parameters:
            payload["top_p"] = api_parameters["top_p"]

        if "presence_penalty" in api_parameters:
            payload["presence_penalty"] = api_parameters["presence_penalty"]

        # Search-related parameters
        if "search_domain_filter" in api_parameters:
            payload["search_domain_filter"] = api_parameters["search_domain_filter"]

        if "search_recency_filter" in api_parameters:
            payload["search_recency_filter"] = api_parameters["search_recency_filter"]

        headers = {
            "Authorization": f"Bearer {settings.perplexity_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            async with httpx.AsyncClient() as client:
                logger.debug("🔍 Perplexity Streaming API Request:")
                logger.debug(f"   URL: {self.base_url}/chat/completions")
                logger.debug(f"   Model: {model_id}")
                logger.debug(f"   Messages: {len(llm_messages)}")

                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"❌ Perplexity Streaming Error: {response.status_code}")
                        yield {
                            "id": f"chatcmpl-{model_id}",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": f"Error: HTTP {response.status_code}"},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        return

                    logger.info(f"✅ Started streaming response from Perplexity {model_id}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix

                            if data_str.strip() == "[DONE]":
                                break

                            try:
                                chunk_data = json.loads(data_str)

                                # Convert Perplexity chunk to OpenAI format
                                if "choices" in chunk_data and chunk_data["choices"]:
                                    choice = chunk_data["choices"][0]
                                    delta = choice.get("delta", {})

                                    yield {
                                        "id": chunk_data.get("id", f"chatcmpl-{model_id}"),
                                        "object": "chat.completion.chunk",
                                        "created": chunk_data.get("created", 0),
                                        "model": model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": delta,
                                                "finish_reason": choice.get("finish_reason"),
                                            }
                                        ],
                                    }

                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse streaming chunk: {e}")
                                continue

                    # Send final chunk if not already sent
                    yield {
                        "id": f"chatcmpl-{model_id}",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }

        except Exception as e:
            logger.error(f"❌ Error streaming from Perplexity {model_id}: {e}")
            yield {
                "id": f"chatcmpl-{model_id}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Streaming error: {str(e)}"},
                        "finish_reason": "stop",
                    }
                ],
            }

    def _convert_perplexity_response(
        self, perplexity_response: Dict[str, Any], model_id: str
    ) -> Dict[str, Any]:
        """Convert Perplexity response format to OpenAI-compatible format"""

        # Research shows Perplexity has different response structure
        # Try to handle both old and new response formats

        if "choices" in perplexity_response:
            # Standard OpenAI-like format
            return perplexity_response
        elif "output" in perplexity_response:
            # New Perplexity format with output array
            output = perplexity_response["output"]
            if output and len(output) > 0:
                message_content = ""
                for item in output:
                    if item.get("type") == "message" and "content" in item:
                        content_array = item["content"]
                        for content_item in content_array:
                            if content_item.get("type") == "text":
                                message_content += content_item.get("text", "")

                # Convert to OpenAI format
                return {
                    "id": perplexity_response.get("id", "unknown"),
                    "object": "chat.completion",
                    "created": int(perplexity_response.get("created_at", 0)),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": message_content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": perplexity_response.get(
                        "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    ),
                }

        # Fallback if format is unexpected
        logger.warning(f"Unknown Perplexity response format: {perplexity_response.keys()}")
        return {
            "id": "unknown",
            "object": "chat.completion",
            "created": 0,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Error: Could not parse Perplexity response format",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
