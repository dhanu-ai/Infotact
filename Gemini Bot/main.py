import base64, dotenv
import os
from google import genai
from google.genai import types
import streamlit as st

# Load environment variables
dotenv.load_dotenv()

# Fetch API key
api_key = os.getenv("GOOGLE_API")
if not api_key:
    raise ValueError("Missing GOOGLE_API key in the environment variables.")

# Pre-instruction for the model
PRE_INSTRUCTION = "You are a helpful assistant. Provide clear, concise, and polite responses."

def generate(user_input):
    # Initialize conversation history in session state if not already done
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=PRE_INSTRUCTION)],
            )
        ]
    
    client = genai.Client(api_key=api_key)
    
    model = "gemini-2.0-pro-exp-02-05"

    # Add new user input to the conversation history
    st.session_state.conversation_history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        )
    )

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    # Generate response with the updated history
    response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=st.session_state.conversation_history,  # Pass full history
        config=generate_content_config,
    ):
        response += chunk.text

    # Append model response to the conversation history
    st.session_state.conversation_history.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=response)],
        )
    )
    
    return response

def main():
    st.set_page_config(page_title="Gemini Tutor", page_icon="📘", layout="wide")
    st.markdown("""
        <style>
            .user-message {
                background-color: #dcf8c6;
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
            }
            .assistant-message {
                background-color: #f1f0f0;
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📘 Gemini Tutor")
        st.write("Ask me anything and I'll try to help you out, just like a personal tutor!")

    st.title("Welcome to Gemini Tutor")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message["role"]
        container_class = "user-message" if role == "user" else "assistant-message"
        st.markdown(f'<div class="{container_class}">{message["content"]}</div>', unsafe_allow_html=True)

    # Accept user input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        response = generate(prompt)
        st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
