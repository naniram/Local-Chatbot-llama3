import streamlit as st
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Custom prompt template to incorporate memory and context
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["history", "input"], 
    template="""You are a helpful AI assistant. 
Previous conversation:
{history}

Current conversation:
Human: {input}
Assistant:"""
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(return_messages=True)

def get_llm_response(user_input):
    """Generate response from local Llama model."""
    # Initialize Ollama with Llama 3.2 model
    llm = Ollama(model="llama3.2")
    
    # Create conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state['memory'],
        prompt=CUSTOM_PROMPT,
        verbose=True
    )
    
    # Get response
    response = conversation.predict(input=user_input)
    
    return response

def main():
    st.title("🤖 Local Llama 3.2 Chatbot")
    
    # Initialize session state
    initialize_session_state()
    
    # Chat input
    user_input = st.chat_input("Enter your message:")
    
    # Display chat history
    for message in st.session_state['chat_history']:
        if message['role'] == 'human':
            st.chat_message('human').write(message['content'])
        else:
            st.chat_message('assistant').write(message['content'])
    
    # Process new message
    if user_input:
        # Display user message
        st.chat_message('human').write(user_input)
        
        # Get AI response
        response = get_llm_response(user_input)
        
        # Display AI response
        st.chat_message('assistant').write(response)
        
        # Update chat history
        st.session_state['chat_history'].append({
            'role': 'human', 
            'content': user_input
        })
        st.session_state['chat_history'].append({
            'role': 'assistant', 
            'content': response
        })
        
        # Rerun to update the chat interface
        st.rerun()

if __name__ == "__main__":
    main()