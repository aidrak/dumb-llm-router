import uuid
import time
import tiktoken
import json
from typing import Dict, Any, List
from fastapi import HTTPException

from clients.base import get_llm_client_and_model_details
from config.models import ChatRequest, Message
from config.settings import settings
from utils.logger import setup_logger
from services.conversation_state import conversation_state_manager

logger = setup_logger(__name__)

class SmartRouter:
    def __init__(self):
        pass
    
    def _get_text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        return ""

    def _detect_heavy_context(self, messages: List[Message]) -> bool:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not get tiktoken encoding, falling back to character count: {e}")
            total_length = sum(len(self._get_text_from_content(msg.content)) for msg in messages)
            return total_length > (settings.token_usage_threshold * 4)

        num_tokens = 0
        for message in messages:
            text_content = self._get_text_from_content(message.content)
            if text_content:
                num_tokens += len(encoding.encode(text_content))
        return num_tokens > settings.token_usage_threshold

    def _has_image_in_current_message(self, message: Message) -> bool:
        """Check if the current message contains an image"""
        if isinstance(message.content, list):
            has_image = any(part.get("type") == "image_url" for part in message.content)
            if has_image:
                logger.info(f"🖼️ FOUND IMAGE in current message!")
            return has_image
        return False

    def _has_image_in_conversation(self, messages: List[Message]) -> bool:
        """Check if any message in the conversation contains an image"""
        for i, message in enumerate(messages):
            if self._has_image_in_current_message(message):
                logger.info(f"🖼️ FOUND IMAGE in message #{i+1}")
                return True
        return False

    def _references_image_content(self, text: str) -> bool:
        """Check if the text references image-related content"""
        image_reference_keywords = [
            "image", "picture", "photo", "screenshot", "device", "list", "show", "display",
            "read", "see", "view", "visible", "shown", "listed", "those", "these",
            "what's in", "what is in", "describe", "caption", "identify", "items",
            "objects", "contents", "details", "information", "from the image",
            "in the picture", "what you see", "analyze this", "tell me about",
            "third item", "first item", "second item", "last item", "bottom", "top",
            "left", "right", "corner", "center", "highlighted", "selected",
            "what's the", "which one", "how many", "count", "number of"
        ]
        
        text_lower = text.lower()
        matched_keywords = [kw for kw in image_reference_keywords if kw in text_lower]
        
        if matched_keywords:
            logger.info(f"🔍 MATCHED IMAGE KEYWORDS: {matched_keywords}")
        
        return len(matched_keywords) > 0

    def _should_use_vision_model(self, messages: List[Message], conversation_state: Dict[str, Any]) -> bool:
        """Determine if we should use a vision-capable model"""
        last_message = messages[-1] if messages else None
        if not last_message:
            return False
        
        logger.info("🔍 CHECKING VISION REQUIREMENTS...")
        
        # Check if current message has an image
        if self._has_image_in_current_message(last_message):
            logger.info("✅ VISION REQUIRED: Current message contains image")
            return True
        
        # Check if conversation has images and current message references them
        if self._has_image_in_conversation(messages):
            last_text = self._get_text_from_content(last_message.content)
            logger.info(f"🔍 CHECKING IMAGE REFERENCES in text: '{last_text}'")
            
            if self._references_image_content(last_text):
                logger.info("✅ VISION REQUIRED: Text references image content")
                return True
            
            # If the message is short and potentially referring to something visual
            if len(last_text.strip()) < 50:
                logger.info("✅ VISION REQUIRED: Short message with images in conversation")
                return True
        
        logger.info("❌ VISION NOT REQUIRED")
        return False

    def _is_research_request(self, text: str) -> bool:
        """Check if the text contains research-related keywords"""
        research_keywords = [
            "research", "look up", "lookup", "find out", "search for", 
            "what is", "who is", "when is", "where is", "how is",
            "tell me about", "get information on", "what are the latest",
            "can you lookup", "please lookup", "can you look up", "please look up"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in research_keywords)

    async def route_and_process(self, chat_request: ChatRequest, auth_header: str) -> Dict[str, Any]:
        settings.reload_if_changed()
        
        messages = chat_request.messages
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        last_message = messages[-1]
        user_prompt_text = self._get_text_from_content(last_message.content)
        
        logger.info(f"🚀 ROUTING REQUEST: '{user_prompt_text[:100]}...'")
        logger.info(f"📊 CONVERSATION LENGTH: {len(messages)} messages")
        
        # Debug: Log message content types
        for i, msg in enumerate(messages):
            content_type = "string" if isinstance(msg.content, str) else f"list[{len(msg.content)}]"
            logger.info(f"📝 Message {i+1} ({msg.role}): {content_type}")
            if isinstance(msg.content, list):
                for j, part in enumerate(msg.content):
                    logger.info(f"   Part {j+1}: {part.get('type', 'unknown')}")
        
        # Get conversation state
        conversation_state = conversation_state_manager.get_or_create_state(messages)

        # Check for vision and heavy context requirements first
        needs_vision = self._should_use_vision_model(messages, conversation_state)
        
        is_research = self._is_research_request(user_prompt_text)
        
        if is_research or conversation_state.perplexity_sticky_count > 0:
            logger.info("🔍 Research request detected, routing to Perplexity.")
            target_model = "Perplexity-Research"
        else:
            target_model = settings.primary_model
        
        logger.info(f"🎯 TARGET MODEL SELECTED: {target_model}")
        
        conversation_state_manager.update_state_after_routing(
            state=conversation_state,
            selected_model=target_model,
            model_tier="research" if is_research else "simple",
            is_research=is_research
        )

        client, model_id, model_type, api_params, system_prompt = get_llm_client_and_model_details(target_model)
        
        if not client or not model_id:
            logger.warning(f"⚠️ Target model '{target_model}' not available, trying fallback.")
            client, model_id, model_type, api_params, system_prompt = get_llm_client_and_model_details(settings.fallback_model)
            if not client or not model_id:
                raise HTTPException(status_code=500, detail="No valid LLM client available")
            target_model = settings.fallback_model

        # CRITICAL DEBUG INFO
        provider_name = getattr(client, 'provider_name', 'unknown')
        logger.info(f"🔧 FINAL ROUTING DECISION:")
        logger.info(f"   📋 Logical Model: {target_model}")
        logger.info(f"   🏭 Provider: {provider_name}")
        logger.info(f"   🤖 Actual Model ID: {model_id}")
        logger.info(f"   🖼️ Vision Required: {needs_vision}")
        
        # Double-check: If vision is required but we're not using Gemini, FORCE Gemini
        if needs_vision and provider_name != 'gemini':
            logger.error(f"🚨 CRITICAL ERROR: Vision required but routing to {provider_name}!")
            logger.error(f"🔧 FORCING GEMINI MODEL...")
            
            # Force to Gemini Pro
            client, model_id, model_type, api_params, system_prompt = get_llm_client_and_model_details(settings.escalation_model)
            if client and getattr(client, 'provider_name', '') == 'gemini':
                target_model = settings.escalation_model
                provider_name = 'gemini'
                logger.info(f"✅ FORCED TO GEMINI: {target_model} -> {model_id}")
            else:
                raise HTTPException(status_code=500, detail="Cannot find Gemini model for vision task!")
        
        try:
            temperature = chat_request.temperature if chat_request.temperature is not None else 0.7
            message_dicts = [msg.dict() for msg in messages]
            
            logger.info(f"🚀 CALLING MODEL: {target_model} ({provider_name}: {model_id})")
            
            # Log the actual request being sent (truncated, excluding base64 images)
            logger.debug(f"📤 REQUEST MESSAGES: {len(message_dicts)} messages")
            for i, msg in enumerate(message_dicts):
                content = msg.get('content', '')
                if isinstance(content, list):
                    # Handle multimodal content
                    parts_summary = []
                    for part in content:
                        if part.get('type') == 'text':
                            text_preview = part.get('text', '')[:50]
                            parts_summary.append(f"text: '{text_preview}...'")
                        elif part.get('type') == 'image_url':
                            parts_summary.append("image: [base64_data]")
                        else:
                            parts_summary.append(f"{part.get('type', 'unknown')}: [data]")
                    content_preview = f"[{', '.join(parts_summary)}]"
                else:
                    content_preview = str(content)[:100]
                logger.debug(f"   Message {i+1}: {content_preview}...")
            
            response = await client.generate_response(
                model_id=model_id,
                messages=message_dicts,
                temperature=temperature,
                api_parameters=api_params,
                system_prompt=system_prompt
            )
            
            # Log response details
            if hasattr(response, 'choices') and response.choices:
                response_preview = response.choices[0].message.content[:100]
                logger.info(f"✅ RESPONSE RECEIVED: '{response_preview}...'")
            elif isinstance(response, dict) and 'choices' in response:
                response_preview = response['choices'][0]['message']['content'][:100]
                logger.info(f"✅ RESPONSE RECEIVED: '{response_preview}...'")
            
            logger.info(f"✅ Generated response using {target_model} ({model_id})")
            return self._format_openai_response(response, target_model)
            
        except Exception as e:
            logger.error(f"❌ Error generating response with {target_model}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    def _format_openai_response(self, llm_response: Any, model_name: str) -> Dict[str, Any]:
        # Check if this is an image generation response
        if isinstance(llm_response, dict) and "data" in llm_response and llm_response.get("object") == "list":
            # This is an image generation response - return it as-is but ensure it's not treated as streaming
            logger.info(f"🖼️ Returning image generation response with {len(llm_response['data'])} images")
            
            # Convert to a format that works better with OpenWebUI streaming
            return {
                "id": llm_response.get("id", f"img-{__import__('uuid').uuid4()}"),
                "object": "chat.completion",  # Change from "list" to "chat.completion"
                "created": llm_response.get("created", int(__import__('time').time())),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I've generated an image based on your request."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10},
                # Include the image data
                "images": llm_response.get("data", [])
            }
        
        # Check if this is an error response
        if isinstance(llm_response, dict) and "error" in llm_response:
            logger.error(f"❌ Returning error response: {llm_response['error']}")
            return llm_response
        
        # Standard chat completion response handling
        content = "An error occurred or no content was generated."
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
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
                usage = {
                    "prompt_tokens": llm_response.get('prompt_eval_count', 0),
                    "completion_tokens": llm_response.get('eval_count', 0),
                    "total_tokens": llm_response.get('prompt_eval_count', 0) + llm_response.get('eval_count', 0)
                }
        
        # Only add model prefix in debug mode and log it
        if settings.log_level.upper() in ["DEBUG"] and not content.startswith(f"{model_name} - "):
            content = f"{model_name} - {content}"
            logger.debug(f"🏷️ ADDED MODEL PREFIX: {model_name}")

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": usage
        }

router = SmartRouter()
