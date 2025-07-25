from fastapi import FastAPI, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
import time
import json

# Import our modular components
from config.models import ChatRequest
from config.settings import settings
from services.router import router
from utils.auth import validate_api_key
from utils.logger import setup_logger, update_log_level
from services.conversation_state import conversation_state_manager

logger = setup_logger(__name__)

app = FastAPI(title="Smart LLM Router", version="2.0.0")

def _clean_request_data_for_logging(req_data):
    """Clean request data to avoid logging massive base64 images"""
    cleaned = req_data.copy()
    
    if 'messages' in cleaned:
        cleaned_messages = []
        for msg in cleaned['messages']:
            cleaned_msg = msg.copy()
            if isinstance(cleaned_msg.get('content'), list):
                cleaned_content = []
                for part in cleaned_msg['content']:
                    if part.get('type') == 'image_url':
                        cleaned_content.append({
                            'type': 'image_url',
                            'image_url': {'url': '[base64_image_data_truncated]'}
                        })
                    else:
                        cleaned_content.append(part)
                cleaned_msg['content'] = cleaned_content
            cleaned_messages.append(cleaned_msg)
        cleaned['messages'] = cleaned_messages
    
    return cleaned

@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """
    Returns the smart router as a single model option.
    Open WebUI uses this endpoint to discover models.
    """
    validate_api_key(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "dumb-llm-router",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "permission": [],
                "root": "dumb-llm-router",
                "parent": None
            }
        ]
    }

@app.get("/models")
async def list_models_alt(authorization: str = Header(None)):
    """Alternative models endpoint"""
    return await list_models(authorization)

@app.post("/v1/chat/completions")
async def smart_route_chat_completions(request: Request, authorization: str = Header(None)):
    """Main chat completions endpoint with smart routing and streaming support"""
    try:
        # Hot-reload settings and update log level
        settings.reload_if_changed()
        update_log_level()
        
        validate_api_key(authorization)
        
        # Parse request and log the cleaned data
        req_data = await request.json()
        logger.debug(f"🔍 RAW REQUEST DATA KEYS: {list(req_data.keys())}")
        
        # Only log cleaned request data in debug mode
        if settings.log_level.upper() == "DEBUG":
            cleaned_data = _clean_request_data_for_logging(req_data)
            logger.debug(f"🔍 CLEANED REQUEST DATA: {cleaned_data}")
        
        chat_request = ChatRequest(**req_data)
        
        # FIXED: Log files at top level instead of in messages
        num_files = len(chat_request.files) if chat_request.files else 0
        logger.debug(f"📄 NUMBER OF FILES: {num_files}")
        
        if chat_request.files:
            for i, file_ref in enumerate(chat_request.files):
                logger.debug(f"📄 File {i+1}: ID={file_ref.id}, Type={file_ref.type}, Name={file_ref.name}")
        
        # Get authorization header for file fetching
        auth_header = request.headers.get("Authorization", "")
        logger.debug(f"🔑 AUTH HEADER LENGTH: {len(auth_header) if auth_header else 0}")
        
        # Check if streaming is requested
        stream = chat_request.stream or req_data.get("stream", False)
        
        if stream:
            logger.info("🌊 Handling streaming request")
            
            # Handle streaming request
            async def streaming_generator():
                try:
                    async for chunk in router.route_and_process_streaming(chat_request, auth_header):
                        yield chunk
                except Exception as e:
                    logger.error(f"❌ Error in streaming generator: {e}")
                    # Send error chunk
                    error_chunk = {
                        "id": "error",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "dumb-llm-router",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: {str(e)}"},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                streaming_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*"
                }
            )
        else:
            logger.info("📄 Handling non-streaming request")
            
            # Route and process the request (non-streaming)
            response = await router.route_and_process(chat_request, auth_header)
            
            return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Critical error in smart routing service: {e}", exc_info=True)
        
        # Check if this was a streaming request for proper error response
        req_data = await request.json() if hasattr(request, '_body') else {}
        is_streaming = req_data.get("stream", False)
        
        if is_streaming:
            # Return streaming error
            async def error_stream():
                error_chunk = {
                    "id": "error",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "dumb-llm-router",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"Critical error: {str(e)}"},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream"
            )
        else:
            # Return regular error
            return JSONResponse(
                content={
                    "error": {
                        "message": f"A critical error occurred in the smart router: {str(e)}",
                        "type": "server_error",
                        "code": "smart_router_fail"
                    }
                },
                status_code=500
            )

@app.get("/debug/conversations")
async def debug_conversations(authorization: str = Header(None)):
    """Debug endpoint to view conversation states"""
    validate_api_key(authorization)
    
    total_conversations = len(conversation_state_manager.states)
    
    if total_conversations == 0:
        return {
            "total_conversations": 0,
            "conversations": []
        }
    
    # Get summary of all conversations
    conversations = []
    for conv_id in list(conversation_state_manager.states.keys())[:10]:  # Limit to 10 for readability
        summary = conversation_state_manager.get_state_summary(conv_id)
        conversations.append(summary)
    
    return {
        "total_conversations": total_conversations,
        "showing": len(conversations),
        "conversations": conversations
    }

@app.get("/debug/conversation/{conversation_id}")
async def debug_specific_conversation(conversation_id: str, authorization: str = Header(None)):
    """Debug endpoint to view a specific conversation state"""
    validate_api_key(authorization)
    
    summary = conversation_state_manager.get_state_summary(conversation_id)
    return summary

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Smart LLM Router v2.0 is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)