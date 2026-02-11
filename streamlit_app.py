import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from graph import create_chatbot_graph, run_conversation
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage



load_dotenv()


@st.cache_resource
def initialize_chatbot():
    """
    Initialize the chatbot graph (cached to avoid recreating on every interaction).
    """
    return create_chatbot_graph()


def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="LangGraph Chatbot", 
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 LangGraph Chatbot")
    st.markdown("A conversational AI with memory powered by LangGraph")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key check
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            st.success("✅ API Key loaded")
        else:
            st.error("❌ API Key not found")
            st.info("Please add your OPENROUTER_API_KEY to the .env file")
            return
        
        # Conversation controls
        st.header("💬 Conversation")
        
        if st.button("🔄 New Conversation", type="primary"):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        # Stats
        if "messages" in st.session_state:
            st.header("📊 Stats")
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
            ai_messages = len([m for m in st.session_state.messages if isinstance(m, AIMessage)])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", total_messages)
                st.metric("You", user_messages)
            with col2:
                st.metric("Assistant", ai_messages)
    
    # Initialize session state
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "app" not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.app = initialize_chatbot()
    
    # Display conversation history
    st.header("💬 Conversation")
    
    # Create a container for messages
    message_container = st.container()
    
    with message_container:
        if not st.session_state.messages:
            st.info("👋 Start a conversation by typing a message below!")
        else:
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to display
        with st.chat_message("user"):
            st.write(user_input)
        
        # Show thinking indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run the conversation through LangGraph
                result = run_conversation(
                    st.session_state.app, 
                    user_input, 
                    st.session_state.conversation_id
                )
                
                # Update session state with the conversation history
                st.session_state.messages = result.get("messages", [])
                
                # Display assistant response
                assistant_response = result.get("assistant_response", "I'm sorry, I couldn't process that.")
                st.write(assistant_response)
        
        # Rerun to update the display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Built by Aaron Gurram using LangGraph and Streamlit")


def display_example_conversations():
    """
    Display example conversations in an expander.
    """
    with st.expander("💡 Example Conversations"):
        st.markdown("""
        **Try these conversation starters:**
        
        🗣️ **Getting to know the assistant:**
        - "Hello! What can you help me with?"
        - "What's your name and what are your capabilities?"
        
        🧠 **Testing memory:**
        - "My name is John"
        - "What's my name?" (should remember from previous message)
        
        🤔 **Problem solving:**
        - "Can you help me plan a weekend trip?"
        - "What are some good Python libraries for data analysis?"
        
        💭 **Creative tasks:**
        - "Write a short story about a robot learning to cook"
        - "Help me brainstorm ideas for a blog post about AI"
        """)


if __name__ == "__main__":
    main()
    display_example_conversations()