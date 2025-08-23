from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FileDataContent(BaseModel):
    content: Optional[str] = None


class FileDataObject(BaseModel):
    data: Optional[FileDataContent] = None


class FileReference(BaseModel):
    type: str
    id: Optional[str] = None  # Images don't have IDs
    name: Optional[str] = None
    file: Optional[FileDataObject] = None  # Text files have this
    url: Optional[str] = None  # Images have this instead
    # Add other fields that might be present
    status: Optional[str] = None
    size: Optional[int] = None
    error: Optional[str] = None
    itemId: Optional[str] = None


from typing import Union


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageUrl(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl


class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    files: Optional[List[FileReference]] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    files: Optional[List[FileReference]] = None  # Add top-level files
    # Handle other OpenWebUI fields that might be present
    stream: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    tool_servers: Optional[List[Any]] = None
    features: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None
    model_item: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    id: Optional[str] = None
    background_tasks: Optional[Dict[str, Any]] = None
