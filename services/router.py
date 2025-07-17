import uuid
import time
from typing import Dict, Any, List
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
                        logger.info(f"🖼️ Vision content detected")
                        return True
        return False

    async def route_and_process(self, chat_request: ChatRequest, auth_header: str) -> Dict[str, Any]:
        messages = chat_request.messages
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        last_message = messages[-1]
        user_prompt_text = self._get_text_from_content(last_message.content)
        
        logger.info(f"🚀 Processing request: '{user_prompt_text[:100]}...'")
        
        # Simple routing logic: always use primary model
        # Gemini with Perplexity will handle search decisions automatically
        target_model = settings.primary_model
        
        # Ensure we have vision capability if needed
        has_vision = self._has_vision_content(messages)
        if has_vision:
            logger.info("🖼️ Vision content detected - using vision-capable model")
        
        logger.info(f"🎯 Selected model: {target_model}")

        # Get client and model details
        client, model_id, model_type, api_params, system_prompt = get_llm_client_and_model_details(target_model)
        
        if not client or not model_id:
            logger.warning(f"⚠️ Target model '{target_model}' not available, trying fallback.")
            client, model_id, model_type, api_params, system_prompt = get_llm_client_and_model_details(settings.fallback_model)
            if not client or not model_id:
                raise HTTPException(status_code=500, detail="No valid LLM client available")
            target_model = settings.fallback_model

        provider_name = getattr(client, 'provider_name', 'unknown')
        logger.info(f"🔧 Using: {target_model} ({provider_name}: {model_id})")

        try:
            temperature = chat_request.temperature if chat_request.temperature is not None else 0.7
            message_dicts = [msg.dict() for msg in messages]
            
            response = await client.generate_response(
                model_id=model_id,
                messages=message_dicts,
                temperature=temperature,
                api_parameters=api_params,
                system_prompt=system_prompt
            )
            
            logger.info(f"✅ Response generated successfully")
            return self._format_openai_response(response, target_model)
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
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
        if isinstance(llm_response, dict) and "data" in llm_response and llm_response.get("object") == "list":
            return {
                "id": llm_response.get("id", f"img-{uuid.uuid4()}"),
                "object": "chat.completion",
                "created": llm_response.get("created", int(time.time())),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I've generated an image based on your request."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10},
                "images": llm_response.get("data", [])
            }
        
        # Handle error responses
        if isinstance(llm_response, dict) and "error" in llm_response:
            return llm_response
        
        # Standard chat completion response
        content = "An error occurred or no content was generated."
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Extract content from response
        if hasattr(llm_response, 'choices') and llm_response.choices:
            content = llm_response.choices[0].message.content
            if hasattr(llm_response, 'usage'):
                usage = {
                    "prompt_tokens": getattr(llm_response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(llm_response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(llm_response.usage, 'total_tokens', 0)
                }
        elif isinstance(llm_response, dict):
            if 'choices' in llm_response:
                content = llm_response['choices'][0]['message']['content']
                usage = llm_response.get('usage', usage)
            elif 'message' in llm_response:
                content = llm_response['message']['content']

        # Add debug prefix only in debug mode
        if settings.log_level.upper() == "DEBUG" and not content.startswith(f"{model_name} - "):
            content = f"{model_name} - {content}"

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": usage
        }

router = SmartRouter()