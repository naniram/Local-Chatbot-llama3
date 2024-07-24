import streamlit as st
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize Ollama
llm = Ollama(model="llama3", 
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def generate_response(user_input):
    # Construct the full context from the message history
    context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    
    # Add the new user input
    prompt = f"{context}\nHuman: {user_input}\nAssistant:"

    # Generate response using Ollama via langchain
    response = llm(prompt)
    
    return response.strip()

def main():
    st.title("Contextual Chatbot with Llama 3")

    initialize_chat()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "human", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("human"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                assistant_response = generate_response(prompt)
            message_placeholder.markdown(assistant_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Add a button to clear the chat history
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()