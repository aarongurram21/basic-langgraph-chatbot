from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    """
    State schema for the chatbot conversation.
    
    This defines the structure of data that flows through the graph nodes.
    """
    # List of messages in the conversation (with memory)
    messages: List[BaseMessage]
    
    # User's current input
    user_input: str
    
    # LLM's response
    assistant_response: str
    
    # Conversation metadata
    conversation_id: str
    
    # Any additional context or metadata
    metadata: Dict[str, Any]


# Alternative approach using add_messages for automatic message handling
class SimpleChatState(TypedDict):
    """
    Simplified state schema using LangGraph's built-in message handling.
    """
    messages: List[BaseMessage] = add_messages  # Automatically handles message appending
    user_input: str
    metadata: Dict[str, Any]