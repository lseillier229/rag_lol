import streamlit as st
import os
import sys
import requests

# Optionnel : si tu veux encore pouvoir lancer en local avec un backend Python dans le repo
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# URL du backend :
# - en local : tu peux exporter BACKEND_URL=http://localhost:8000
# - en cluster K8s : http://rag-backend-service:8000 (Service Kubernetes)
BACKEND_URL = os.getenv("BACKEND_URL", "http://rag-backend-service:8000")
# BACKEND_URL="http://localhost:8001"

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def call_backend(question: str) -> str:
    """
    Appelle l'API FastAPI du backend pour obtenir la réponse.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/answer",
            json={"question": question},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", "Erreur : réponse vide du backend.")
    except Exception as e:
        return f"Erreur lors de l'appel au backend : {e}"


def main():
    st.set_page_config(page_title="LoL RAG Chatbot", page_icon="🧠")
    st.title("🧠💬 LoL RAG Chatbot (CamemBERT + FAISS + rerank lexical)")

    init_session_state()

    # Affichage de l'historique
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
                answer = call_backend(user_input)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
