# Fully configurable Gemini client with Perplexity search integration
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

import httpx
import json
from typing import Dict, Any, List, Optional, AsyncGenerator  # ADD AsyncGenerator HERE
from clients.base import BaseLLMClient
from config.settings import settings
from utils.logger import setup_logger
import base64

logger = setup_logger(__name__)

class GeminiWithPerplexityClient(BaseLLMClient):
    def __init__(self):
        super().__init__("gemini_perplexity")
    
    def initialize_client(self) -> bool:
        if not settings.gemini_api_key or not genai:
            logger.warning("GEMINI_API_KEY not set or genai SDK not available.")
            return False
        
        if not settings.perplexity_api_key:
            logger.warning("PERPLEXITY_API_KEY not set. Search function will not be available.")
            # Don't return False here - continue without search functionality
        
        try:
            self.client = genai.Client(api_key=settings.gemini_api_key)
            logger.info("✅ Gemini client with configurable Perplexity search initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            return False
    
    def supports_streaming(self) -> bool:
        """Gemini with Perplexity supports streaming"""
        return True
    
    async def _call_perplexity_search(self, query: str, perplexity_config: Dict[str, Any]) -> str:
        """Call Perplexity API with configurable parameters"""
        if not settings.perplexity_api_key:
            return "Search unavailable: Perplexity API key not configured"
        
        # Extract configuration with defaults
        model = perplexity_config.get("model", "sonar-pro")
        endpoint = perplexity_config.get("endpoint", "https://api.perplexity.ai/chat/completions")
        max_tokens = perplexity_config.get("max_tokens", 2000)
        temperature = perplexity_config.get("temperature", 0.1)
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant. Provide comprehensive and accurate information, citing your sources."},
                {"role": "user", "content": query}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "search_focus": "academic"
        }
        
        # Add optional search parameters
        if "search_domain_filter" in perplexity_config:
            payload["search_domain_filter"] = perplexity_config["search_domain_filter"]
        
        if "search_recency_filter" in perplexity_config:
            payload["search_recency_filter"] = perplexity_config["search_recency_filter"]
        
        if "search_after_date_filter" in perplexity_config:
            payload["search_after_date_filter"] = perplexity_config["search_after_date_filter"]
        
        if "search_before_date_filter" in perplexity_config:
            payload["search_before_date_filter"] = perplexity_config["search_before_date_filter"]
        
        headers = {
            "Authorization": f"Bearer {settings.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            timeout = perplexity_config.get("timeout", 30.0)
            async with httpx.AsyncClient() as client:
                logger.debug(f"🔍 Calling Perplexity {model} with query: {query[:100]}...")
                
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Handle both response formats
                if "choices" in data and data["choices"]:
                    search_result = data["choices"][0]["message"]["content"]
                    
                    # Add sources if available
                    if "references" in data and data["references"]:
                        sources = "\n\n**Sources:**\n"
                        for ref in data["references"]:
                            sources += f"- {ref['url']}\n"
                        search_result += sources
                        
                elif "output" in data and data["output"]:
                    # Handle Perplexity's custom format
                    output = data["output"]
                    search_result = ""
                    for item in output:
                        if item.get("type") == "message" and "content" in item:
                            content_array = item["content"]
                            for content_item in content_array:
                                if content_item.get("type") == "text":
                                    search_result += content_item.get("text", "")
                else:
                    search_result = "Search completed but no results found"
                
                logger.info(f"✅ Perplexity search completed using {model}")
                return search_result
                    
        except Exception as e:
            logger.error(f"❌ Perplexity search failed: {e}")
            return f"Search failed: {str(e)}"
    
    def _create_search_function_declaration(self, perplexity_config: Dict[str, Any]) -> types.FunctionDeclaration:
        """Create configurable search function declaration for Gemini"""
        function_name = perplexity_config.get("function_name", "search_web")
        function_description = perplexity_config.get(
            "function_description", 
            "Search the web for current information, recent events, news, facts, or data that may not be in my training data."
        )
        
        # Build parameter schema
        properties = {
            "query": types.Schema(
                type=types.Type.STRING,
                description="The search query to find relevant information"
            )
        }
        
        # Add domain filter parameter if configured
        if "search_domain_filter" in perplexity_config:
            properties["domain_filter"] = types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Optional list of domains to restrict search to (e.g., ['reddit.com', 'stackoverflow.com'])"
            )
        
        return types.FunctionDeclaration(
            name=function_name,
            description=function_description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties=properties,
                required=["query"]
            )
        )
    
    def _convert_openai_content_to_gemini(self, content: Any) -> List[Any]:
        """Convert OpenAI-style content to Gemini format"""
        if isinstance(content, str):
            return [content]
        
        if isinstance(content, list):
            gemini_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        gemini_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image/"):
                            try:
                                header, base64_data = image_url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                image_bytes = base64.b64decode(base64_data)
                                
                                image_part = types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=mime_type
                                )
                                gemini_parts.append(image_part)
                                logger.debug(f"✓ Converted image: {mime_type}")
                            except Exception as e:
                                logger.error(f"❌ Failed to convert image: {e}")
                                gemini_parts.append("(Image conversion failed)")
                        else:
                            gemini_parts.append("(External image URL not supported)")
                else:
                    gemini_parts.append(str(part))
            return gemini_parts
        
        return [str(content)]
    
    def _build_gemini_contents(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> List[Any]:
        """Convert OpenAI messages to Gemini contents format"""
        if not messages:
            return [system_prompt if system_prompt else "Hello"]
        
        all_images = []
        conversation_context = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                content_parts = self._convert_openai_content_to_gemini(content)
                text_parts = []
                
                for part in content_parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    else:
                        all_images.append(part)
                
                if text_parts:
                    user_text = " ".join(text_parts)
                    conversation_context.append(f"User: {user_text}")
                    
            elif role == "assistant":
                assistant_text = content
                if isinstance(assistant_text, str):
                    # Clean up model prefixes
                    if " - " in assistant_text:
                        assistant_text = assistant_text.split(" - ", 1)[1]
                    conversation_context.append(f"Assistant: {assistant_text}")
        
        final_contents = []
        
        if system_prompt:
            final_contents.append(system_prompt)
        
        if conversation_context:
            # Keep last 10 exchanges to maintain context
            recent_context = conversation_context[-10:]
            context_text = "\n".join(recent_context)
            final_contents.append(f"Conversation context:\n{context_text}")
        
        # Add all images from conversation
        final_contents.extend(all_images)
        
        # Add current question for emphasis
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
        
        # Add search function if enabled
        enable_perplexity_search = api_parameters.get("enable_perplexity_search", False)
        perplexity_config = api_parameters.get("perplexity_config", {})
        
        if enable_perplexity_search and settings.perplexity_api_key and perplexity_config:
            search_tool = types.Tool(
                function_declarations=[self._create_search_function_declaration(perplexity_config)]
            )
            if "config" not in generate_params:
                generate_params["config"] = types.GenerateContentConfig()
            generate_params["config"].tools = [search_tool]
            
            perplexity_model = perplexity_config.get("model", "sonar-pro")
            logger.debug(f"✓ Enabled Perplexity search using {perplexity_model}")
        
        try:
            gemini_contents = self._build_gemini_contents(messages, system_prompt)
            
            # Initial response
            response = self.client.models.generate_content(
                model=model_id,
                contents=gemini_contents,
                **generate_params
            )
            
            # Check if function calls were made
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    # Check for function calls
                    function_calls = []
                    text_content = ""
                    
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call is not None:
                            function_calls.append(part.function_call)
                        elif hasattr(part, 'text') and part.text is not None:
                            text_content += part.text
                    
                    # Process function calls
                    if function_calls:
                        function_responses = []
                        
                        for func_call in function_calls:
                            if func_call is None:
                                continue
                                
                            function_name = perplexity_config.get("function_name", "search_web")
                            if hasattr(func_call, 'name') and func_call.name == function_name:
                                if hasattr(func_call, 'args'):
                                    args = func_call.args
                                    query = args.get("query", "") if args else ""
                                    
                                    # Use domain filter from function call or config
                                    domain_filter = None
                                    if args:
                                        domain_filter = args.get("domain_filter") or perplexity_config.get("search_domain_filter")
                                    else:
                                        domain_filter = perplexity_config.get("search_domain_filter")
                                        
                                    if domain_filter:
                                        perplexity_config = {**perplexity_config, "search_domain_filter": domain_filter}
                                    
                                    if query:
                                        logger.info(f"🔍 Gemini requested search: {query}")
                                        search_result = await self._call_perplexity_search(query, perplexity_config)
                                        
                                        function_responses.append(
                                            types.Part(
                                                function_response=types.FunctionResponse(
                                                    name=function_name,
                                                    response={"result": search_result}
                                                )
                                            )
                                        )
                                    else:
                                        logger.warning("🔍 Function call received but no query found")
                                else:
                                    logger.warning("🔍 Function call received but no args found")
                        
                        # Generate final response with search results
                        if function_responses:
                            # Add function responses to conversation
                            updated_contents = gemini_contents + [function_responses]
                            
                            final_response = self.client.models.generate_content(
                                model=model_id,
                                contents=updated_contents,
                                **generate_params
                            )
                            
                            logger.info(f"✅ Generated enhanced response with search results")
                            return {
                                "choices": [{"message": {"content": final_response.text}}],
                                "model": model_id,
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                            }
            
            # Regular response without function calls
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
        
        # Add search function if enabled
        enable_perplexity_search = api_parameters.get("enable_perplexity_search", False)
        perplexity_config = api_parameters.get("perplexity_config", {})
        
        if enable_perplexity_search and settings.perplexity_api_key and perplexity_config:
            search_tool = types.Tool(
                function_declarations=[self._create_search_function_declaration(perplexity_config)]
            )
            if "config" not in generate_params:
                generate_params["config"] = types.GenerateContentConfig()
            generate_params["config"].tools = [search_tool]
            
            perplexity_model = perplexity_config.get("model", "sonar-pro")
            logger.debug(f"✓ Enabled Perplexity search for streaming using {perplexity_model}")

        try:
            gemini_contents = self._build_gemini_contents(messages, system_prompt)
            
            # SIMPLIFIED APPROACH: Handle function calls in non-streaming mode first
            # This avoids the complexity of streaming with function calls
            initial_response = self.client.models.generate_content(
                model=model_id,
                contents=gemini_contents,
                **generate_params
            )
            
            # Check if function calls were made
            function_calls_made = False
            final_content = ""
            
            if hasattr(initial_response, 'candidates') and initial_response.candidates:
                candidate = initial_response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    
                    # Check for both function calls and text
                    function_calls = []
                    text_parts = []
                    
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call is not None:
                            function_calls.append(part.function_call)
                            function_calls_made = True
                        elif hasattr(part, 'text') and part.text is not None:
                            text_parts.append(part.text)
                    
                    # If we have function calls, process them
                    if function_calls_made and enable_perplexity_search:
                        logger.info("🔍 Processing search request before streaming")
                        
                        # Process search requests
                        function_responses = []
                        for func_call in function_calls:
                            if func_call is None:
                                continue
                                
                            function_name = perplexity_config.get("function_name", "search_web")
                            if hasattr(func_call, 'name') and func_call.name == function_name:
                                if hasattr(func_call, 'args'):
                                    args = func_call.args
                                    query = args.get("query", "") if args else ""
                                    
                                    if query:
                                        logger.info(f"🔍 Gemini requested search: {query}")
                                        search_result = await self._call_perplexity_search(query, perplexity_config)
                                        
                                        function_responses.append(
                                            types.Part(
                                                function_response=types.FunctionResponse(
                                                    name=function_name,
                                                    response={"result": search_result}
                                                )
                                            )
                                        )
                        
                        # Generate final response with search results
                        if function_responses:
                            updated_contents = gemini_contents + [function_responses]
                            
                            # Now stream the final response
                            stream = self.client.models.generate_content_stream(
                                model=model_id,
                                contents=updated_contents,
                                **generate_params
                            )
                            
                            logger.info(f"✅ Started streaming enhanced response with search results")
                            
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
                            return
                    
                    # If no function calls or they failed, use any existing text
                    if text_parts:
                        final_content = " ".join(text_parts)
            
            # If we have final content from the initial response, stream it
            if final_content:
                # Simulate streaming by chunking the response
                chunk_size = 50
                for i in range(0, len(final_content), chunk_size):
                    chunk_content = final_content[i:i + chunk_size]
                    
                    yield {
                        "id": f"chatcmpl-{model_id}",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_content},
                            "finish_reason": None
                        }]
                    }
                    
                    # Small delay for natural streaming feel
                    import asyncio
                    await asyncio.sleep(0.05)
            else:
                # Fallback: regular streaming without function calls
                logger.info(f"✅ Started streaming response from Gemini {model_id}")
                
                stream = self.client.models.generate_content_stream(
                    model=model_id,
                    contents=gemini_contents,
                    **generate_params
                )
                
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
