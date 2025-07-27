import streamlit as st
import requests
import os
import json
import time 
from dotenv import load_dotenv

load_dotenv()

RAG_BACKEND_URL = os.getenv('RAG_BACKEND_URL')


st.set_page_config(page_title="Changi airport RAG Chatbot", page_icon="💬")
st.title("Whats' you query? ")


if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_chat_history():
    """Clears the chat history in the session state."""
    st.session_state.messages = []

st.sidebar.button("Clear Chat", on_click=clear_chat_history)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def stream_data(text_response):
    """
    Simulates streaming by yielding words from the response with a delay.
    """
    for word in text_response.split(" "):
        yield word + " "
        time.sleep(0.05) # Small delay to simulate streaming
    # Ensure a final newline or space if needed to prevent cursor issues
    yield "" 


if prompt := st.chat_input("Hi! How may i help you? "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    payload = {"query": prompt}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = "No response found from the RAG model." # Default in case of error
            try:
                response = requests.post(RAG_BACKEND_URL, json=payload)
                response.raise_for_status()

                rag_response_data = response.json()
                full_response = rag_response_data.get("answer", full_response)
                
 
                st.write_stream(stream_data(full_response))

            except requests.exceptions.ConnectionError:
                full_response = "Error: Could not connect to the RAG backend. Please check the URL and ensure the backend is running."
                st.error(full_response)
            except requests.exceptions.Timeout:
                full_response = "Error: The request to the RAG backend timed out."
                st.error(full_response)
            except requests.exceptions.RequestException as e:
                full_response = f"An unexpected error occurred during the API request: {e}"
                st.error(full_response)
            except json.JSONDecodeError:
                full_response = "Error: Received an invalid JSON response from the RAG backend."
                st.error(full_response)
            except Exception as e:
                full_response = f"An unexpected error occurred: {e}"
                st.error(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
