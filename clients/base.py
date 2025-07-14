from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
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
    async def generate_response(self, 
                              model_id: str, 
                              messages: List[Dict[str, Any]], 
                              temperature: float,
                              api_parameters: Dict[str, Any],
                              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate a response using the LLM"""
        pass
    
    def is_available(self) -> bool:
        """Check if the client is available and initialized"""
        return self.client is not None

def get_llm_client_and_model_details(logical_model_name: str) -> Tuple[Optional[BaseLLMClient], Optional[str], Optional[str], Optional[Dict], Optional[str]]:
    """
    Looks up model details from settings and returns the appropriate
    client object, model_id, model_type, api_parameters, and system_prompt.
    """
    model_details = settings.model_configs.get(logical_model_name)
    if not model_details:
        print(f"Error: Model '{logical_model_name}' not found in model configurations.")
        return None, None, None, None, None
    
    provider = model_details.get("provider")
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
    
    if not provider or not model_id_for_api:
        print(f"Error: Incomplete configuration for model '{logical_model_name}'. Missing provider or model_id.")
        return None, None, None, None, None
    
    # Import and initialize the appropriate client
    if provider == "openai":
        from clients.openai_client import OpenAIClient
        client = OpenAIClient()
    elif provider == "ollama":
        from clients.ollama_client import OllamaClient
        client = OllamaClient()
    elif provider == "gemini":
        from clients.gemini_client import GeminiClient
        client = GeminiClient()
    elif provider == "anthropic":
        from clients.anthropic_client import AnthropicClient
        client = AnthropicClient()
    elif provider == "perplexity":
        from clients.perplexity_client import PerplexityClient
        client = PerplexityClient()
    elif provider == "gemini_image":
        from clients.gemini_image_client import GeminiImageClient
        client = GeminiImageClient()
    else:
        print(f"Error: Unsupported provider '{provider}' for model '{logical_model_name}'.")
        return None, None, None, None, None
    
    if not client.initialize_client():
        print(f"Warning: {provider.title()} client failed to initialize for model '{logical_model_name}'.")
        return None, None, None, None, None
    
    return client, model_id_for_api, model_type, api_parameters, system_prompt_content