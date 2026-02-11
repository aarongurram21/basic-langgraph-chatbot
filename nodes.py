import os
from typing import Dict, Any
from langchain_community.chat_models import ChatOpenRouter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import ChatState


class ChatNodes:
    """
    Contains all the node functions for the chatbot graph.
    """
    
    def __init__(self):
        # Initialize the LLM with your OpenRouter API key
        self.llm = ChatOpenRouter(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="mistralai/mistral-7b-instruct",
            temperature=0.7,
            max_tokens=1000
        )
    
    def process_user_input(self, state: ChatState) -> Dict[str, Any]:
        """
        Node to process and validate user input.
        """
        user_input = state.get("user_input", "").strip()
        
        if not user_input:
            return {
                "user_input": "",
                "assistant_response": "Please provide a message to continue our conversation."
            }
        
        # Add user message to conversation history
        messages = state.get("messages", [])
        
        # Add system message if this is the first message
        if not messages:
            system_msg = SystemMessage(content="""You are a helpful and friendly AI assistant. 
            You have access to the conversation history and should provide contextual, 
            intelligent responses. Be concise but informative, and maintain a conversational tone.""")
            messages.append(system_msg)
        
        # Add the user's message
        user_message = HumanMessage(content=user_input)
        messages.append(user_message)
        
        return {
            "messages": messages,
            "user_input": user_input
        }
    
    def generate_response(self, state: ChatState) -> Dict[str, Any]:
        """
        Node to generate LLM response based on conversation history.
        """
        try:
            messages = state.get("messages", [])
            
            if not messages:
                return {
                    "assistant_response": "Hello! How can I help you today?"
                }
            
            # Generate response using the LLM
            response = self.llm.invoke(messages)
            assistant_response = response.content
            
            # Add AI response to messages
            ai_message = AIMessage(content=assistant_response)
            messages.append(ai_message)
            
            return {
                "messages": messages,
                "assistant_response": assistant_response
            }
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            return {
                "assistant_response": error_msg,
                "messages": state.get("messages", []) + [AIMessage(content=error_msg)]
            }
    
    def update_metadata(self, state: ChatState) -> Dict[str, Any]:
        """
        Node to update conversation metadata.
        """
        metadata = state.get("metadata", {})
        messages = state.get("messages", [])
        
        # Update message count and other stats
        metadata.update({
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
            "ai_messages": len([m for m in messages if isinstance(m, AIMessage)]),
            "last_updated": "now"  # In a real app, you'd use datetime
        })
        
        return {
            "metadata": metadata
        }


# Conditional logic functions
def should_process_input(state: ChatState) -> str:
    """
    Determines if we should process user input or skip to response generation.
    """
    user_input = state.get("user_input", "").strip()
    return "process" if user_input else "skip"


def should_continue(state: ChatState) -> str:
    """
    Determines if the conversation should continue.
    In this basic version, we always continue.
    """
    return "continue"