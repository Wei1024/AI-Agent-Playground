# streamlit_app.py
import streamlit as st
from agent import get_agent_response, parse_agent_response

st.title("Agentic AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from agent and parse it as a single string
    chat_result = get_agent_response(st.session_state.messages)
    final_answer = parse_agent_response(chat_result)

    # Display final answer with URLs included
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    with st.chat_message("assistant"):
        st.markdown(final_answer)
