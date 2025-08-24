"""
Gemini RAG Implementation
Handles file processing and RAG functionality using Google's native Files API
Based on the manifold implementation with simplified architecture
"""
import asyncio
import base64
import io
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import xxhash
    from aiocache.backends.memory import SimpleMemoryCache
    from aiocache.serializers import NullSerializer
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError as e:
    print(f"Warning: Missing dependencies for Gemini RAG: {e}")
    genai = None
    types = None
    xxhash = None

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FilesAPIError(Exception):
    """Custom exception for Files API errors"""
    pass


class UploadStatusManager:
    """Manages status updates for concurrent file uploads"""

    def __init__(self, event_emitter=None):
        self.event_emitter = event_emitter
        self.queue = asyncio.Queue()
        self.total_uploads_expected = 0
        self.uploads_completed = 0
        self.finalize_received = False
        self.is_active = False

    async def run(self) -> None:
        """Main manager loop"""
        while not (
            self.finalize_received
            and self.total_uploads_expected == self.uploads_completed
        ):
            msg = await self.queue.get()
            msg_type = msg[0]

            if msg_type == "REGISTER_UPLOAD":
                self.is_active = True
                self.total_uploads_expected += 1
                await self._emit_progress_update()
            elif msg_type == "COMPLETE_UPLOAD":
                self.uploads_completed += 1
                await self._emit_progress_update()
            elif msg_type == "FINALIZE":
                self.finalize_received = True

            self.queue.task_done()

        logger.debug("UploadStatusManager finished")

    async def _emit_progress_update(self) -> None:
        """Emit progress update"""
        if not self.is_active:
            return

        is_done = (
            self.total_uploads_expected > 0
            and self.uploads_completed == self.total_uploads_expected
        )

        if is_done:
            message = f"Upload complete. {self.uploads_completed} file(s) processed."
        else:
            message = f"Uploading file {self.uploads_completed + 1} of {self.total_uploads_expected}..."

        logger.info(message)


class FilesAPIManager:
    """Manages file uploads and caching with Google Files API"""

    def __init__(self, client: genai.Client, event_emitter=None):
        if not genai or not types:
            raise ImportError("google-genai package required for FilesAPIManager")
        if not xxhash:
            raise ImportError("xxhash package required for FilesAPIManager")

        self.client = client
        self.event_emitter = event_emitter
        self.file_cache = SimpleMemoryCache(serializer=NullSerializer())
        self.id_hash_cache = SimpleMemoryCache(serializer=NullSerializer())
        self.upload_locks: Dict[str, asyncio.Lock] = {}

    async def get_or_upload_file(
        self,
        file_bytes: bytes,
        mime_type: str,
        owui_file_id: Optional[str] = None,
        status_queue: Optional[asyncio.Queue] = None,
    ) -> types.File:
        """Get or upload file using content-addressable caching"""

        # Get content hash
        content_hash = await self._get_content_hash(file_bytes, owui_file_id)

        # Check cache first
        cached_file = await self.file_cache.get(content_hash)
        if cached_file:
            logger.debug(f"Cache hit for file hash {content_hash}")
            return cached_file

        # Acquire lock for this file
        lock = self.upload_locks.setdefault(content_hash, asyncio.Lock())
        async with lock:
            # Double-check cache after acquiring lock
            cached_file = await self.file_cache.get(content_hash)
            if cached_file:
                logger.debug(f"Cache hit after lock for file hash {content_hash}")
                return cached_file

            # Try to recover existing file
            deterministic_name = f"files/owui-v1-{content_hash}"
            try:
                file = await self.client.aio.files.get(name=deterministic_name)
                if file.name:
                    logger.debug(f"Recovered existing file: {deterministic_name}")
                    active_file = await self._poll_for_active_state(file.name)
                    ttl_seconds = self._calculate_ttl(active_file.expiration_time)
                    await self.file_cache.set(content_hash, active_file, ttl=ttl_seconds)
                    return active_file

            except genai_errors.ClientError as e:
                if e.code == 403:  # File not found
                    logger.info(f"File not found on server: {deterministic_name}")
                else:
                    logger.error(f"Error checking file status: {e}")
                    raise FilesAPIError(f"Failed to check file status: {e}")

            # Upload new file
            return await self._upload_and_process_file(
                content_hash, file_bytes, mime_type, deterministic_name, status_queue
            )

    async def _get_content_hash(self, file_bytes: bytes, owui_file_id: Optional[str]) -> str:
        """Get content hash with caching optimization"""
        if owui_file_id:
            cached_hash = await self.id_hash_cache.get(owui_file_id)
            if cached_hash:
                return cached_hash

        # Compute hash
        content_hash = xxhash.xxh64(file_bytes).hexdigest()

        if owui_file_id:
            await self.id_hash_cache.set(owui_file_id, content_hash)

        return content_hash

    def _calculate_ttl(self, expiration_time: Optional[datetime]) -> Optional[float]:
        """Calculate TTL in seconds"""
        if not expiration_time:
            return None

        now_utc = datetime.now(timezone.utc)
        if expiration_time <= now_utc:
            return 0

        return (expiration_time - now_utc).total_seconds()

    async def _upload_and_process_file(
        self,
        content_hash: str,
        file_bytes: bytes,
        mime_type: str,
        deterministic_name: str,
        status_queue: Optional[asyncio.Queue] = None,
    ) -> types.File:
        """Upload and process new file"""

        if status_queue:
            await status_queue.put(("REGISTER_UPLOAD",))

        logger.info(f"Uploading new file: {deterministic_name}")

        try:
            file_io = io.BytesIO(file_bytes)
            upload_config = types.UploadFileConfig(
                name=deterministic_name, mime_type=mime_type
            )
            uploaded_file = await self.client.aio.files.upload(
                file=file_io, config=upload_config
            )

            if not uploaded_file.name:
                raise FilesAPIError("File upload did not return a file name")

            # Wait for file to become active
            if uploaded_file.state == types.FileState.ACTIVE:
                active_file = uploaded_file
            else:
                active_file = await self._poll_for_active_state(uploaded_file.name)

            # Cache the file
            ttl_seconds = self._calculate_ttl(active_file.expiration_time)
            await self.file_cache.set(content_hash, active_file, ttl=ttl_seconds)

            logger.info(f"File upload completed: {active_file.name}")
            return active_file

        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise FilesAPIError(f"Upload failed: {e}")
        finally:
            if status_queue:
                await status_queue.put(("COMPLETE_UPLOAD",))
            # Clean up lock
            if content_hash in self.upload_locks:
                del self.upload_locks[content_hash]

    async def _poll_for_active_state(
        self, file_name: str, timeout: int = 60, poll_interval: int = 1
    ) -> types.File:
        """Poll file until it becomes ACTIVE"""
        end_time = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < end_time:
            try:
                file = await self.client.aio.files.get(name=file_name)

                if file.state == types.FileState.ACTIVE:
                    return file

                if file.state == types.FileState.FAILED:
                    error_msg = f"File processing failed: {file_name}"
                    if file.error:
                        error_msg += f" - {file.error.message}"
                    raise FilesAPIError(error_msg)

                logger.debug(f"File {file_name} still processing...")
                await asyncio.sleep(poll_interval)

            except Exception as e:
                raise FilesAPIError(f"Polling failed for {file_name}: {e}")

        raise FilesAPIError(f"File {file_name} did not become active within {timeout}s")


class GeminiContentBuilder:
    """Builds Gemini content from messages and files"""

    def __init__(self, files_api_manager: FilesAPIManager, use_files_api: bool = True):
        self.files_api_manager = files_api_manager
        self.use_files_api = use_files_api

    async def build_contents_from_files(
        self, files: List[Dict[str, Any]]
    ) -> List[types.Part]:
        """Build content parts from file references"""
        if not files:
            return []

        logger.info(f"🔄 Starting Gemini RAG processing for {len(files)} files")

        # Set up status manager
        status_manager = UploadStatusManager(self.files_api_manager.event_emitter)
        manager_task = asyncio.create_task(status_manager.run())

        # Process files concurrently
        tasks = []
        for file_ref in files:
            task = self._process_file_reference(file_ref, status_manager.queue)
            tasks.append(task)

        logger.debug(f"Processing {len(tasks)} files concurrently")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Signal completion
        await status_manager.queue.put(("FINALIZE",))
        await manager_task

        # Filter successful results
        parts = []
        for i, result in enumerate(results):
            if isinstance(result, types.Part):
                parts.append(result)
                logger.debug(f"✅ Successfully processed file {i+1}")
            elif isinstance(result, Exception):
                logger.error(f"❌ File {i+1} processing error: {result}")
            else:
                logger.warning(f"⚠️ File {i+1} returned no content (fallback to Open WebUI RAG)")

        if parts:
            logger.info(f"✅ Gemini RAG processed {len(parts)} out of {len(files)} files successfully")
        else:
            logger.info("⚠️ No files were processed by Gemini RAG - falling back to Open WebUI RAG")

        return parts

    async def _process_file_reference(
        self, file_ref: Dict[str, Any], status_queue: asyncio.Queue
    ) -> Optional[types.Part]:
        """Process a single file reference"""
        try:
            file_type = file_ref.get("type", "")
            file_id = file_ref.get("id")

            if file_type == "file" and file_id:
                # Handle regular file
                return await self._process_regular_file(file_id, status_queue)
            elif file_type == "image":
                # Handle image
                return await self._process_image_file(file_ref, status_queue)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return None

        except Exception as e:
            logger.error(f"Error processing file reference: {e}")
            return None

    async def _process_regular_file(
        self, file_id: str, status_queue: asyncio.Queue
    ) -> Optional[types.Part]:
        """Process regular file by ID"""
        try:
            # Get file data
            file_bytes, mime_type = await self._get_file_data(file_id)
            if not file_bytes or not mime_type:
                return None

            if self.use_files_api:
                # Use Files API
                gemini_file = await self.files_api_manager.get_or_upload_file(
                    file_bytes=file_bytes,
                    mime_type=mime_type,
                    owui_file_id=file_id,
                    status_queue=status_queue,
                )
                return types.Part(
                    file_data=types.FileData(
                        file_uri=gemini_file.uri,
                        mime_type=gemini_file.mime_type,
                    )
                )
            else:
                # Send as raw bytes
                return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        except Exception as e:
            logger.error(f"Error processing regular file {file_id}: {e}")
            return None

    async def _process_image_file(
        self, file_ref: Dict[str, Any], status_queue: asyncio.Queue
    ) -> Optional[types.Part]:
        """Process image file"""
        try:
            image_url = file_ref.get("url", "")

            if image_url.startswith("data:image"):
                # Handle data URI
                match = re.match(r"data:(image/\w+);base64,(.+)", image_url)
                if not match:
                    logger.error("Invalid data URI for image")
                    return None

                mime_type, base64_data = match.group(1), match.group(2)
                file_bytes = base64.b64decode(base64_data)

                if self.use_files_api:
                    # Use Files API
                    gemini_file = await self.files_api_manager.get_or_upload_file(
                        file_bytes=file_bytes,
                        mime_type=mime_type,
                        status_queue=status_queue,
                    )
                    return types.Part(
                        file_data=types.FileData(
                            file_uri=gemini_file.uri,
                            mime_type=gemini_file.mime_type,
                        )
                    )
                else:
                    # Send as raw bytes
                    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            else:
                logger.warning(f"Unsupported image URL format: {image_url[:50]}...")
                return None

        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            return None

    async def _get_file_data(self, file_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Get file data from Open WebUI storage - simplified version"""
        try:
            # This would normally integrate with Open WebUI's file system
            # For now, we'll implement a basic version that attempts to fetch from a common path
            # In production, this should use Open WebUI's Files API

            # Placeholder implementation - return None to trigger fallback to Open WebUI RAG
            logger.info(f"File data retrieval not fully implemented for file_id: {file_id}")
            logger.info("Falling back to Open WebUI RAG for file processing")
            return None, None

        except Exception as e:
            logger.error(f"Error retrieving file data for {file_id}: {e}")
            return None, None
