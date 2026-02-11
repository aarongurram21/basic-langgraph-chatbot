import os
import uuid
from dotenv import load_dotenv
from graph import create_chatbot_graph, run_conversation


def main():
    """
    Main CLI interface for the chatbot.
    """
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please set your API key in the .env file.")
        return
    
    # Create the chatbot graph
    print("Initializing LangGraph Chatbot...")
    app = create_chatbot_graph()
    
    # Generate a unique conversation ID
    conversation_id = str(uuid.uuid4())
    
    print("\n" + "="*50)
    print("🤖 LangGraph Chatbot - CLI Interface")
    print("="*50)
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'clear' to start a new conversation.")
    print("Type 'stats' to see conversation statistics.")
    print("-"*50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting!")
                break
            
            if user_input.lower() == 'clear':
                conversation_id = str(uuid.uuid4())
                print("🔄 Conversation cleared. Starting fresh!")
                continue
            
            if user_input.lower() == 'stats':
                # This would require storing state externally for CLI
                print("📊 Stats feature would require persistent storage in CLI mode.")
                continue
            
            if not user_input:
                print("⚠️  Please enter a message.")
                continue
            
            # Run the conversation
            print("🤖 Assistant: ", end="", flush=True)
            result = run_conversation(app, user_input, conversation_id)
            
            # Display the response
            print(result['assistant_response'])
            
            # Optionally show metadata
            metadata = result.get('metadata', {})
            if metadata:
                print(f"   (Messages in conversation: {metadata.get('total_messages', 0)})")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


def test_conversation():
    """
    Test function to verify the chatbot works.
    """
    load_dotenv()
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: API key not found. Please check your .env file.")
        return
    
    app = create_chatbot_graph()
    conversation_id = "test_conversation"
    
    test_messages = [
        "Hello! What's your name?",
        "What's the weather like today?",
        "Can you remember what I asked you first?"
    ]
    
    print("🧪 Running test conversation...")
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Test {i} ---")
        print(f"User: {message}")
        
        result = run_conversation(app, message, conversation_id)
        print(f"Assistant: {result['assistant_response']}")
        
        metadata = result.get('metadata', {})
        print(f"Total messages: {metadata.get('total_messages', 0)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_conversation()
    else:
        main()