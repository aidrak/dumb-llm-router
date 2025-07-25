# clients/gemini_client.py
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from typing import Dict, Any, List, Optional, AsyncGenerator
from clients.base import BaseLLMClient
from config.settings import settings
from utils.logger import setup_logger
import base64
import re
import json

logger = setup_logger(__name__)

class GeminiClient(BaseLLMClient):
    def __init__(self):
        super().__init__("gemini")
    
    def initialize_client(self) -> bool:
        if not settings.gemini_api_key or not genai:
            logger.warning("GEMINI_API_KEY not set or genai SDK not available. Gemini models will not be available.")
            return False
        
        try:
            self.client = genai.Client(api_key=settings.gemini_api_key)
            logger.info("✅ Gemini client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            return False
    
    def supports_streaming(self) -> bool:
        """Gemini supports streaming"""
        return True
    
    def _convert_openai_content_to_gemini(self, content: Any) -> List[Any]:
        """Convert OpenAI-style content to Gemini format using the new SDK"""
        if isinstance(content, str):
            # Simple text content
            return [content]
        
        if isinstance(content, list):
            gemini_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        gemini_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        # Convert OpenAI image format to Gemini format
                        image_url = part.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image/"):
                            # Extract base64 data
                            try:
                                # Format: data:image/png;base64,<base64_data>
                                header, base64_data = image_url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                
                                # Decode base64 to bytes
                                image_bytes = base64.b64decode(base64_data)
                                
                                # Use the correct new SDK method
                                image_part = types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=mime_type
                                )
                                gemini_parts.append(image_part)
                                logger.debug(f"✓ Converted image to Gemini Part: {mime_type}")
                            except Exception as e:
                                logger.error(f"❌ Failed to convert image: {e}")
                                gemini_parts.append("(Image conversion failed)")
                        else:
                            # External URL - not supported in this example
                            gemini_parts.append("(External image URL not supported)")
                else:
                    # Fallback for unexpected format
                    gemini_parts.append(str(part))
            return gemini_parts
        
        # Fallback
        return [str(content)]
    
    def _build_gemini_contents(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> List[Any]:
        """Convert OpenAI messages to Gemini contents format with full conversation context"""
        if not messages:
            return [system_prompt if system_prompt else "Hello"]
        
        # For Gemini, we'll build a comprehensive context that includes the conversation history
        # but focuses on the most recent exchange while preserving images from the entire conversation
        
        # Collect all images from the conversation
        all_images = []
        conversation_context = []
        
        for i, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                # Process user message content
                content_parts = self._convert_openai_content_to_gemini(content)
                text_parts = []
                
                for part in content_parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    else:
                        # This is an image Part - collect it
                        all_images.append(part)
                
                if text_parts:
                    user_text = " ".join(text_parts)
                    conversation_context.append(f"User: {user_text}")
                    
            elif role == "assistant":
                # Include assistant responses (clean up the model prefixes)
                assistant_text = content
                if isinstance(assistant_text, str):
                    # Remove model name prefixes like "Flash-Research - "
                    if " - " in assistant_text:
                        assistant_text = assistant_text.split(" - ", 1)[1]
                    conversation_context.append(f"Assistant: {assistant_text}")
        
        # Build the final content
        final_contents = []
        
        # Add system prompt if provided
        if system_prompt:
            final_contents.append(system_prompt)
        
        # Add conversation context (limit to last few exchanges to avoid token limits)
        if conversation_context:
            # Keep last 10 exchanges to maintain context but not overwhelm
            recent_context = conversation_context[-10:]
            context_text = "\n".join(recent_context)
            final_contents.append(f"Conversation context:\n{context_text}")
        
        # Add all images from the conversation
        final_contents.extend(all_images)
        
        # If the last message was just text with no images, add it again for emphasis
        last_message = messages[-1]
        if last_message.get("role") == "user":
            last_content = last_message.get("content", "")
            if isinstance(last_content, str):
                final_contents.append(f"Current question: {last_content}")
        
        return final_contents
    
    async def generate_response(self, model_id: str, messages: List[Dict[str, Any]], 
                              temperature: float, api_parameters: Dict[str, Any],
                              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        
        # Build generation config
        generation_config = {}
        if not api_parameters.get("exclude_temperature", False):
            generation_config["temperature"] = temperature
        
        generation_config_params = api_parameters.get("generation_config", {})
        generation_config.update(generation_config_params)
        
        # Build generate_content parameters
        generate_params = {}
        if generation_config:
            generate_params["config"] = types.GenerateContentConfig(
                temperature=generation_config.get("temperature"),
                max_output_tokens=generation_config.get("max_output_tokens")
            )
        
        # Handle search functionality
        enable_search = api_parameters.get("enable_google_search", False)
        if enable_search and types:
            try:
                search_tool = types.Tool(google_search={})
                if "config" not in generate_params:
                    generate_params["config"] = types.GenerateContentConfig()
                generate_params["config"].tools = [search_tool]
                logger.info(f"✓ Enabled Google Search for {model_id}")
            except Exception as e:
                logger.warning(f"❌ Google Search setup failed for {model_id}: {e}")
        
        # Convert messages to Gemini format
        try:
            gemini_contents = self._build_gemini_contents(messages, system_prompt)
            logger.debug(f"🔍 Converted {len(messages)} messages to Gemini contents")
            
            response = self.client.models.generate_content(
                model=model_id,
                contents=gemini_contents,
                **generate_params
            )
            logger.info(f"✅ Received response from Gemini {model_id}")
            
            return {
                "choices": [{"message": {"content": response.text}}],
                "model": model_id,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        except Exception as e:
            logger.error(f"❌ Error calling Gemini {model_id}: {e}")
            raise
    
    async def generate_streaming_response(self, model_id: str, messages: List[Dict[str, Any]], 
                                        temperature: float, api_parameters: Dict[str, Any],
                                        system_prompt: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        
        # Build generation config
        generation_config = {}
        if not api_parameters.get("exclude_temperature", False):
            generation_config["temperature"] = temperature
        
        generation_config_params = api_parameters.get("generation_config", {})
        generation_config.update(generation_config_params)
        
        # Build generate_content parameters
        generate_params = {}
        if generation_config:
            generate_params["config"] = types.GenerateContentConfig(
                temperature=generation_config.get("temperature"),
                max_output_tokens=generation_config.get("max_output_tokens")
            )
        
        # Handle search functionality
        enable_search = api_parameters.get("enable_google_search", False)
        if enable_search and types:
            try:
                search_tool = types.Tool(google_search={})
                if "config" not in generate_params:
                    generate_params["config"] = types.GenerateContentConfig()
                generate_params["config"].tools = [search_tool]
                logger.info(f"✓ Enabled Google Search for {model_id}")
            except Exception as e:
                logger.warning(f"❌ Google Search setup failed for {model_id}: {e}")
        
        # Convert messages to Gemini format
        try:
            gemini_contents = self._build_gemini_contents(messages, system_prompt)
            logger.debug(f"🔍 Converted {len(messages)} messages to Gemini contents for streaming")
            
            # Use streaming method
            stream = self.client.models.generate_content_stream(
                model=model_id,
                contents=gemini_contents,
                **generate_params
            )
            
            logger.info(f"✅ Started streaming response from Gemini {model_id}")
            
            # Process streaming chunks
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield {
                        "id": f"chatcmpl-{model_id}",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk.text},
                            "finish_reason": None
                        }]
                    }
            
            # Send final chunk
            yield {
                "id": f"chatcmpl-{model_id}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
        except Exception as e:
            logger.error(f"❌ Error streaming from Gemini {model_id}: {e}")
            # Yield error chunk
            yield {
                "id": f"chatcmpl-{model_id}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"Error: {str(e)}"},
                    "finish_reason": "stop"
                }]
            }