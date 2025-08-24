from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from config.settings import settings


class BaseLLMClient(ABC):
    """Base class for all LLM clients"""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.client = None

    @abstractmethod
    def initialize_client(self) -> bool:
        """Initialize the client and return True if successful"""
        pass

    @abstractmethod
    async def generate_response(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        api_parameters: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response using the LLM"""
        pass

    async def generate_streaming_response(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        api_parameters: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a streaming response using the LLM. Override in subclasses that support streaming."""
        # Default implementation: fall back to non-streaming
        response = await self.generate_response(
            model_id, messages, temperature, api_parameters, system_prompt
        )

        # Convert to streaming format
        yield {
            "id": "stream_fallback",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": response["choices"][0]["message"]["content"]},
                    "finish_reason": None,
                }
            ],
        }

        # Send final chunk
        yield {
            "id": "stream_fallback",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    def supports_streaming(self) -> bool:
        """Override in subclasses to indicate streaming support"""
        return False

    def is_available(self) -> bool:
        """Check if the client is available and initialized"""
        return self.client is not None


def get_llm_client_and_model_details(
    logical_model_name: str,
) -> Tuple[Optional[BaseLLMClient], Optional[str], Optional[str], Optional[Dict], Optional[str]]:
    """
    Looks up model details from settings and returns the appropriate
    client object, model_id, model_type, api_parameters, and system_prompt.
    """
    model_details = settings.model_configs.get(logical_model_name)
    if not model_details:
        print(f"Error: Model '{logical_model_name}' not found in model configurations.")
        return None, None, None, None, None

    model_id_for_api = model_details.get("model_id")
    model_type = model_details.get("type", "chat")
    api_parameters = model_details.get("parameters", {})

    # Handle system prompt
    system_prompt_content = model_details.get("system_prompt")
    system_prompt_file = model_details.get("system_prompt_file")

    if system_prompt_file:
        file_content = settings.read_system_prompt_from_file(system_prompt_file)
        if file_content:
            system_prompt_content = file_content

    if not system_prompt_content:
        system_prompt_content = "You are a helpful AI assistant."

    if not model_id_for_api:
        print(
            f"Error: Incomplete configuration for model '{logical_model_name}'. Missing model_id."
        )
        return None, None, None, None, None

    client: Optional[BaseLLMClient] = None
    # Map model names directly to clients (no provider field needed)
    if logical_model_name == "working_model":
        from clients.gemini_advanced_client import GeminiAdvancedClient
        client = GeminiAdvancedClient()
    elif logical_model_name == "searching_model":
        from clients.perplexity_client import PerplexityClient
        client = PerplexityClient()
    else:
        print(f"Error: Unknown model '{logical_model_name}'. Expected 'working_model' or 'searching_model'.")
        return None, None, None, None, None

    if not client.initialize_client():
        print(
            f"Warning: Client failed to initialize for model '{logical_model_name}'."
        )
        return None, None, None, None, None

    return client, model_id_for_api, model_type, api_parameters, system_prompt_content
