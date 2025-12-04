import json
from pathlib import Path
from typing import List, Dict, Any
import unicodedata
import numpy as np
import streamlit as st
import torch
from transformers import CamembertModel, CamembertTokenizerFast
import faiss
import re
import os
from groq import Groq  # <-- NEW


DATA_DIR = Path("data/champions")
EMBED_MODEL_NAME = "camembert-base"
MAX_LENGTH = 256
TOP_K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle Groq (change le nom si tu veux un autre modèle)
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "et", "en",
    "au", "aux", "pour", "avec", "que", "qui", "quel", "quelle",
    "quels", "quelles", "est", "sont", "champion", "champions", "donne",
    "moi", "je", "un", "une", "des"
}


# ============================
# CHARGEMENT DU MODELE
# ============================

@st.cache_resource
def load_camembert():
    tokenizer = CamembertTokenizerFast.from_pretrained(EMBED_MODEL_NAME)
    model = CamembertModel.from_pretrained(EMBED_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def get_groq_client():
    """
    Crée un client Groq une seule fois (cache Streamlit).
    Nécessite GROQ_API_KEY dans les variables d'environnement.
    """
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY n'est pas définie dans les variables d'environnement."
        )
    return Groq(api_key=api_key)


# ============================
# CHARGEMENT DES CHAMPIONS
# ============================

def build_text_from_champion(champ: Dict[str, Any]) -> str:
    """
    Construit un texte descriptif optimisé pour le RAG.
    """
    parts = []

    summary = champ.get("summary")
    if summary:
        parts.append(summary)

    name = champ["name"]
    title = champ.get("title", "")
    region = champ.get("region", "Inconnue")
    roles = champ.get("roles") or []
    lanes = champ.get("lanes") or []
    tags = champ.get("tags") or []

    header = f"{name}, {title}".strip()
    parts.append(header)
    parts.append(f"Région : {region}")
    if roles:
        parts.append("Rôles : " + ", ".join(roles))
    if lanes:
        parts.append("Lanes : " + ", ".join(lanes))
    if tags:
        parts.append("Tags : " + ", ".join(tags))

    if champ.get("lore_short"):
        parts.append("Description : " + champ["lore_short"])

    abilities = champ.get("abilities") or []
    if abilities:
        parts.append("Compétences :")
        for ab in abilities:
            parts.append(
                f"- {ab.get('key', '')} | {ab.get('name', '')} : {ab.get('description', '')}"
            )

    return "\n".join(parts)


def load_champion_docs() -> List[Dict[str, Any]]:
    """
    Charge tous les JSON du dossier data/champions
    et construit une liste de "documents" prêts pour l'index.
    """
    docs = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            champ = json.load(f)

        text = build_text_from_champion(champ)
        docs.append(
            {
                "id": path.stem,
                "name": champ.get("name", path.stem),
                "raw": champ,
                "text": text,
            }
        )
    return docs


def embed_texts(
    texts: List[str],
    tokenizer: CamembertTokenizerFast,
    model: CamembertModel,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """
    Calcule un embedding pour chaque texte. On utilise le token [CLS]
    (position 0) puis L2-normalisation pour pouvoir utiliser un IndexFlatIP
    (approx cosinus).
    """
    with torch.no_grad():
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        outputs = model(**batch)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        return cls_embeddings.cpu().numpy()


@st.cache_resource
def build_index():
    """
    Construit l'index vectoriel FAISS une seule fois et le met en cache.
    """
    tokenizer, model = load_camembert()
    docs = load_champion_docs()

    if not docs:
        raise RuntimeError(f"Aucun champion trouvé dans {DATA_DIR} (mets tes JSON ici)")

    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts, tokenizer, model)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "docs": docs,
        "index": index,
        "embeddings": embeddings,
    }


def search_similar_docs(query: str, top_k: int = TOP_K):
    store = build_index()
    tokenizer = store["tokenizer"]
    model = store["model"]
    docs = store["docs"]
    index = store["index"]

    query_emb = embed_texts([query], tokenizer, model)
    k = min(top_k, len(docs))
    scores, indices = index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = docs[idx]
        results.append(
            {
                "score": float(score),
                "doc": doc,
            }
        )
    return results


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenisation très simple.
    """
    text = normalize_text(text)
    tokens = re.split(r"[^a-z0-9]+", text)
    cleaned = []
    for t in tokens:
        if not t:
            continue
        if t in STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        cleaned.append(t)
        if t.endswith("s") and len(t) > 2:
            cleaned.append(t[:-1])
    return cleaned


def rerank_results(question: str, results: List[Dict[str, Any]],
                   alpha=1.0, beta=0.3, gamma=0.8):
    """
    Combine :
      - score FAISS (dense)
      - overlap lexical sur tout le texte
      - overlap sur les métadonnées (roles, lanes, tags, region)
    """
    q_tokens = set(simple_tokenize(question))

    reranked = []
    for r in results:
        doc = r["doc"]
        doc_text = doc["text"]

        doc_tokens = set(simple_tokenize(doc_text))
        lexical_overlap = len(q_tokens & doc_tokens)

        meta_strings = []

        region = doc.get("region")
        if region:
            meta_strings.append(region)

        for field in ("roles", "lanes", "tags"):
            val = doc.get("raw", {}).get(field) or doc.get(field)
            if isinstance(val, str):
                meta_strings.append(val)
            elif isinstance(val, list):
                meta_strings.extend(val)

        meta_tokens = set()
        for s in meta_strings:
            meta_tokens.update(simple_tokenize(str(s)))

        meta_overlap = len(q_tokens & meta_tokens)

        combined = alpha * r["score"] + beta * lexical_overlap + gamma * meta_overlap

        reranked.append(
            {
                **r,
                "lexical_overlap": lexical_overlap,
                "meta_overlap": meta_overlap,
                "combined_score": combined,
            }
        )

    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return reranked


# ============================
# LLM GROQ POUR LA RÉPONSE
# ============================

def generate_lol_answer_with_groq(question: str,
                                  ranked_results: List[Dict[str, Any]],
                                  max_docs: int = 5) -> str:
    """
    Utilise Groq pour générer une réponse finale à partir
    de la question et des meilleurs documents RAG.
    """
    client = get_groq_client()

    # On construit un contexte lisible à partir des top documents
    context_chunks = []
    for r in ranked_results[:max_docs]:
        doc = r["doc"]
        name = doc["name"]
        text = doc["text"]
        context_chunks.append(f"### Champion : {name}\n{text}")

    context = "\n\n".join(context_chunks)

    system_msg = (
        "Tu es un expert de League of Legends. "
        "Tu réponds en français, de façon claire et structurée. "
        "Tu t'appuies uniquement sur le contexte fourni (champions et leurs descriptions). "
        "Si tu n'as pas assez d'information, tu l'indiques honnêtement."
    )

    user_msg = f"""
Question utilisateur :
{question}

Contexte (fiches de champions) :
{context}

Consignes :
- Réponds directement à la question.
- Cite les champions concernés.
- Sois concis mais informatif.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


def answer_question(question: str, top_k: int = TOP_K) -> str:
    """
    Pipeline RAG :
      1) Retrieval dense (CamemBERT + FAISS)
      2) Rerank lexical
      3) Appel LLM Groq pour générer la réponse finale
         + debug (scores, overlaps).
    """
    raw_results = search_similar_docs(question, top_k)
    if not raw_results:
        return "Je ne trouve rien dans ma base de connaissances LoL pour cette question 😅"

    results = rerank_results(question, raw_results)

    # On garde le meilleur champion pour le debug
    best = results[0]
    best_name = best["doc"]["name"]

    debug_lines = []
    for r in results[:3]:
        debug_lines.append(
            f"{r['doc']['name']} "
            f"(combined={r['combined_score']:.2f}, "
            f"FAISS={r['score']:.2f}, "
            f"lex={r['lexical_overlap']}, meta={r['meta_overlap']})"
        )
    debug_block = "\n".join(debug_lines)

    try:
        llm_answer = generate_lol_answer_with_groq(question, results)
        final_answer = (
            f"{llm_answer}\n\n"
            f"---\n"
            f"_Champion le plus probable selon le RAG : **{best_name}**_\n\n"
            f"_Debug (pour le projet) :_\n{debug_block}"
        )
    except Exception as e:
        # Fallback si Groq plante
        final_answer = (
            f"Le champion qui correspond le mieux à ta question est : **{best_name}**.\n\n"
            f"_Debug (Groq a échoué : {e})_\n"
            f"_Scores :_\n{debug_block}"
        )

    return final_answer


# ============================
# STREAMLIT APP
# ============================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(page_title="LoL RAG Chatbot", page_icon="x")
    st.title("LoL RAG Chatbot (CamemBERT + FAISS + Groq LLM)")
    st.caption(
        "Pose des questions sur les champions de League of Legends. "
        "Le bot utilise un RAG : embeddings CamemBERT + FAISS, reranking lexical, "
        "puis un LLM Groq pour générer la réponse finale."
    )

    init_session_state()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Pose ta question (ex : Quel champion est un mage d'Ionia ?)"
    )
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
