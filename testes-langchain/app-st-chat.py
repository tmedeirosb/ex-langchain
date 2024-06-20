#https://docs.streamlit.io/develop/api-reference/chat
import streamlit as st
import numpy as np

st.title("Chat com seu PDF")

messages = st.container()
if prompt := st.chat_input("Say something"):
    messages.chat_message("user").write(prompt)
    messages.chat_message("assistant").write(f"Echo: {prompt}")

message2 = st.chat_message("assistant")
message2.write("Hello human")
message2.bar_chart(np.random.randn(30, 3))

with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))