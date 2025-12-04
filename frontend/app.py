import streamlit as st
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.rag_core import answer_question

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(page_title="LoL RAG Chatbot", page_icon="🧠")
    st.title("🧠💬 LoL RAG Chatbot (CamemBERT + FAISS + rerank lexical)")

    init_session_state()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Pose ta question (ex : Quel champion est un mage d'Ionia ?)")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Je fouille dans le Grimoire de Runeterra..."):
                answer = answer_question(user_input)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
