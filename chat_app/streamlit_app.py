import streamlit as st
import requests

st.title("Chatbot Interface")

# Define the API endpoint
API_ENDPOINT = "http://127.0.0.1:5000/ask"

# Create a chat history list
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display the chat history
for message in st.session_state.chat_history:
    if message['sender'] == 'user':
        st.write(f"You: {message['message']}")
    else:
        st.write(f"Bot: {message['message']}")

# Text input for user message
user_message = st.text_input("Type your message:")

if st.button("Send"):
    if user_message:
        # Save user's message to chat history
        st.session_state.chat_history.append({
            'sender': 'user',
            'message': user_message
        })

        # Send user's message to the backend API
        response = requests.post(API_ENDPOINT, json={'question': question})
        
        # Get the bot's response from API
        bot_response = response.json().get('response', 'Sorry, I could not understand that.')

        # Save bot's response to chat history
        st.session_state.chat_history.append({
            'sender': 'bot',
            'message': bot_response
        })

        # Refresh the page to display the updated chat history
        st.experimental_rerun()
