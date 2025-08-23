import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.models import Message
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ConversationState:
    """Track conversation state for intelligent routing decisions"""

    conversation_id: str
    active_model_tier: str = "simple"  # simple, hard, escalation
    last_model_used: Optional[str] = None
    has_vision_content: bool = False
    vision_sticky_count: int = 0  # Messages to stay on vision model
    heavy_context_active: bool = False
    heavy_context_sticky_count: int = 0
    perplexity_sticky_count: int = 0
    message_count_since_upgrade: int = 0
    last_activity: float = 0.0
    topic_keywords: List[str] = None

    def __post_init__(self):
        if self.topic_keywords is None:
            self.topic_keywords = []
        self.last_activity = time.time()


class ConversationStateManager:
    """Manage conversation states with automatic cleanup"""

    def __init__(self, max_conversations: int = 1000, cleanup_after_hours: int = 24):
        self.states: Dict[str, ConversationState] = {}
        self.max_conversations = max_conversations
        self.cleanup_after_seconds = cleanup_after_hours * 3600
        self.last_cleanup = time.time()

    def _generate_conversation_id(self, messages: List[Message]) -> str:
        """Generate a consistent conversation ID from message history"""
        if not messages:
            return "empty_conversation"

        # Use hash of all message content as fingerprint
        full_conversation_content = "".join(
            [self._get_text_from_content(msg.content) for msg in messages]
        )

        return hashlib.md5(full_conversation_content.encode()).hexdigest()[:16]

    def _get_text_from_content(self, content: Any) -> str:
        """Extract text content from message content"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        return ""

    def _extract_topic_keywords(self, messages: List[Message]) -> List[str]:
        """Extract key topic words from recent messages for topic change detection"""
        if not messages:
            return []

        # Get last 3 messages for topic context
        recent_messages = messages[-3:]
        text_content = " ".join(
            [
                self._get_text_from_content(msg.content)
                for msg in recent_messages
                if msg.role == "user"
            ]
        ).lower()

        # Simple keyword extraction (you could enhance this with NLP)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "this",
            "that",
            "these",
            "those",
        }

        words = [word.strip(".,!?;:()[]{}\"'") for word in text_content.split()]
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        return keywords[:10]  # Keep top 10 keywords

    def _detect_topic_change(
        self, messages: List[Message], current_state: ConversationState
    ) -> bool:
        """Detect if the conversation topic has significantly changed"""
        last_message_content = self._get_text_from_content(messages[-1].content).lower()

        # Explicit topic change indicators
        topic_change_indicators = [
            "new topic",
            "different question",
            "now let's",
            "switching to",
            "change subject",
            "moving on",
            "next question",
            "unrelated",
            "completely different",
            "new request",
            "forget about",
            "ignore that",
        ]

        if any(indicator in last_message_content for indicator in topic_change_indicators):
            logger.info("Detected explicit topic change indicator")
            return True

        # Keyword-based topic drift detection
        current_keywords = self._extract_topic_keywords(messages)
        stored_keywords = current_state.topic_keywords

        if stored_keywords and current_keywords:
            # Calculate keyword overlap
            overlap = len(set(current_keywords) & set(stored_keywords))
            total_unique = len(set(current_keywords) | set(stored_keywords))

            if total_unique > 0:
                overlap_ratio = overlap / total_unique
                if overlap_ratio < 0.3:  # Less than 30% keyword overlap
                    logger.info(
                        f"Detected topic change via keyword analysis: {overlap_ratio:.2f} overlap"
                    )
                    return True

        return False

    def _cleanup_old_conversations(self):
        """Remove old conversation states to prevent memory bloat"""
        if time.time() - self.last_cleanup < 3600:  # Only cleanup once per hour
            return

        current_time = time.time()
        expired_conversations = [
            conv_id
            for conv_id, state in self.states.items()
            if current_time - state.last_activity > self.cleanup_after_seconds
        ]

        for conv_id in expired_conversations:
            del self.states[conv_id]

        # If we're still over the limit, remove oldest conversations
        if len(self.states) > self.max_conversations:
            sorted_conversations = sorted(self.states.items(), key=lambda x: x[1].last_activity)

            excess_count = len(self.states) - self.max_conversations
            for conv_id, _ in sorted_conversations[:excess_count]:
                del self.states[conv_id]

        self.last_cleanup = current_time

        if expired_conversations:
            logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")

    def get_or_create_state(self, messages: List[Message]) -> ConversationState:
        """Get existing conversation state or create new one"""
        self._cleanup_old_conversations()

        conversation_id = self._generate_conversation_id(messages)

        if conversation_id in self.states:
            state = self.states[conversation_id]
            state.last_activity = time.time()

            # Check for topic change
            if self._detect_topic_change(messages, state):
                logger.info(
                    f"Topic change detected for conversation {conversation_id}, resetting state"
                )
                # Reset sticky states but keep some context
                state.vision_sticky_count = 0
                state.heavy_context_sticky_count = 0
                state.perplexity_sticky_count = 0
                state.message_count_since_upgrade = 0
                state.active_model_tier = "simple"
                # Don't reset has_vision_content immediately - let it decay naturally
        else:
            # Create new conversation state
            state = ConversationState(
                conversation_id=conversation_id,
                topic_keywords=self._extract_topic_keywords(messages),
            )
            self.states[conversation_id] = state
            logger.info(f"Created new conversation state: {conversation_id}")

        # Update topic keywords
        state.topic_keywords = self._extract_topic_keywords(messages)

        return state

    def update_state_after_routing(
        self,
        state: ConversationState,
        selected_model: str,
        model_tier: str,
        had_vision: bool = False,
        had_heavy_context: bool = False,
        is_research: bool = False,
    ):
        """Update state after routing decision is made"""
        state.last_model_used = selected_model
        state.last_activity = time.time()

        # Handle vision stickiness
        if had_vision:
            state.has_vision_content = True
            state.vision_sticky_count = 3  # Stay sticky for 3 messages
            state.active_model_tier = "hard"
        elif state.vision_sticky_count > 0:
            state.vision_sticky_count -= 1
            if state.vision_sticky_count == 0:
                state.has_vision_content = False

        # Handle heavy context stickiness
        if had_heavy_context:
            state.heavy_context_active = True
            state.heavy_context_sticky_count = 2  # Stay sticky for 2 messages
            state.active_model_tier = "hard"
        elif state.heavy_context_sticky_count > 0:
            state.heavy_context_sticky_count -= 1
            if state.heavy_context_sticky_count == 0:
                state.heavy_context_active = False

        # Handle perplexity stickiness
        if is_research:
            state.perplexity_sticky_count = 3  # Stay sticky for 3 messages
        elif state.perplexity_sticky_count > 0:
            state.perplexity_sticky_count -= 1

        # Update tier and message count
        if model_tier != state.active_model_tier:
            state.message_count_since_upgrade = 0
            state.active_model_tier = model_tier
        else:
            state.message_count_since_upgrade += 1

    def should_stick_to_vision_model(self, state: ConversationState) -> bool:
        """Check if we should stick to vision model based on state"""
        return state.has_vision_content and state.vision_sticky_count > 0

    def should_stick_to_heavy_model(self, state: ConversationState) -> bool:
        """Check if we should stick to heavy model based on state"""
        return (
            state.heavy_context_active and state.heavy_context_sticky_count > 0
        ) or self.should_stick_to_vision_model(state)

    def get_state_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get a summary of conversation state for debugging"""
        if conversation_id not in self.states:
            return {"status": "not_found"}

        state = self.states[conversation_id]
        return {
            "conversation_id": state.conversation_id,
            "active_model_tier": state.active_model_tier,
            "last_model_used": state.last_model_used,
            "has_vision_content": state.has_vision_content,
            "vision_sticky_count": state.vision_sticky_count,
            "heavy_context_active": state.heavy_context_active,
            "heavy_context_sticky_count": state.heavy_context_sticky_count,
            "perplexity_sticky_count": state.perplexity_sticky_count,
            "message_count_since_upgrade": state.message_count_since_upgrade,
            "topic_keywords": state.topic_keywords[:5],  # First 5 keywords
            "total_conversations_tracked": len(self.states),
        }


# Global instance
conversation_state_manager = ConversationStateManager()
