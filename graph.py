from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import ChatState
from nodes import ChatNodes, should_process_input, should_continue


def create_chatbot_graph():
    """
    Creates and returns the chatbot LangGraph.
    """
    # Initialize nodes
    nodes = ChatNodes()
    
    # Create the graph
    workflow = StateGraph(ChatState)
    
    # Add nodes to the graph
    workflow.add_node("process_input", nodes.process_user_input)
    workflow.add_node("generate_response", nodes.generate_response)
    workflow.add_node("update_metadata", nodes.update_metadata)
    
    # Define the graph flow
    workflow.set_entry_point("process_input")
    
    # Add edges
    workflow.add_edge("process_input", "generate_response")
    workflow.add_edge("generate_response", "update_metadata")
    workflow.add_edge("update_metadata", END)
    
    # Alternative with conditional logic (commented out for simplicity)
    # workflow.add_conditional_edges(
    #     "process_input",
    #     should_process_input,
    #     {
    #         "process": "generate_response",
    #         "skip": END
    #     }
    # )
    
    # Add memory/checkpointing for conversation persistence
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app


def run_conversation(app, user_input: str, conversation_id: str = "default"):
    """
    Run a single conversation turn.
    
    Args:
        app: Compiled LangGraph application
        user_input: User's message
        conversation_id: Unique identifier for the conversation thread
        
    Returns:
        dict: Updated state with assistant response
    """
    # Define the initial state
    initial_state = {
        "user_input": user_input,
        "messages": [],
        "assistant_response": "",
        "conversation_id": conversation_id,
        "metadata": {}
    }
    
    # Configure for conversation persistence
    config = {
        "configurable": {
            "thread_id": conversation_id
        }
    }
    
    # Invoke the graph
    final_state = app.invoke(initial_state, config=config)
    
    return final_state


if __name__ == "__main__":
    # Test the graph
    app = create_chatbot_graph()
    
    # Test conversation
    print("Testing LangGraph Chatbot...")
    
    # First message
    result = run_conversation(app, "Hello! What's your name?", "test_conversation")
    print(f"User: Hello! What's your name?")
    print(f"Assistant: {result['assistant_response']}")
    print(f"Total messages: {result['metadata']['total_messages']}")
    
    # Follow-up message (should remember context)
    result = run_conversation(app, "What did I just ask you?", "test_conversation")
    print(f"\nUser: What did I just ask you?")
    print(f"Assistant: {result['assistant_response']}")
    print(f"Total messages: {result['metadata']['total_messages']}")