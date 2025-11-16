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


# ==============
# CONFIG GLOBALE
# ==============

DATA_DIR = Path("data/champions")   # dossier où tu as mis les JSON
EMBED_MODEL_NAME = "camembert-base"  # modèle pour les embeddings (retriever)
MAX_LENGTH = 256
TOP_K = 20  # on récupère un peu plus large, puis on rerank
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# petit set de stopwords FR pour le score lexical
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


# ============================
# CHARGEMENT DES CHAMPIONS
# ============================

def build_text_from_champion(champ: Dict[str, Any]) -> str:
    """
    Construit un texte descriptif optimisé pour le RAG :
    on commence par un résumé très informatif (summary),
    puis on ajoute les détails.
    """
    parts = []

    # 1) Résumé très important
    summary = champ.get("summary")
    if summary:
        parts.append(summary)

    # 2) Infos structurées
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

    # 3) Lore + spells (comme avant)
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


# ============================
# EMBEDDINGS CAMEMBERT
# ============================

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


# ============================
# INDEX FAISS (RETRIEVAL)
# ============================

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
    index = faiss.IndexFlatIP(dim)  # sur vecteurs normalisés => équivalent cosinus
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
    k = min(top_k, len(docs))  # important
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



# ============================
# RERANK LEXICAL (style IR)
# ============================

def normalize_text(text: str) -> str:
    # minuscule + suppression des accents (é -> e, î -> i, etc.)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenisation très simple :
      - lowercase
      - suppression des accents
      - split sur tout ce qui n'est pas alphanumérique
      - suppression des stopwords
      - gestion basique du pluriel (tanks -> tank, supports -> support, etc.)
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
        # gestion très simple des pluriels en "s"
        if t.endswith("s") and len(t) > 2:
            cleaned.append(t[:-1])
    return cleaned



def rerank_results(question: str, results: List[Dict[str, Any]], alpha=1.0, beta=0.3, gamma=0.8):
    """
    Combine :
      - score FAISS (dense)
      - overlap lexical sur tout le texte
      - overlap sur les métadonnées (roles, lanes, tags, region)

    alpha : poids du score dense
    beta  : poids de l'overlap lexical brut
    gamma : poids de l'overlap sur les métadonnées
    """
    q_tokens = set(simple_tokenize(question))

    reranked = []
    for r in results:
        doc = r["doc"]
        doc_text = doc["text"]

        # 1) Overlap lexical global
        doc_tokens = set(simple_tokenize(doc_text))
        lexical_overlap = len(q_tokens & doc_tokens)

        # 2) Overlap sur les méta (roles, lanes, tags, region)
        meta_strings = []

        # region
        region = doc.get("region")
        if region:
            meta_strings.append(region)

        # roles, lanes, tags
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

        # 3) Score combiné
        combined = alpha * r["score"] + beta * lexical_overlap + gamma * meta_overlap

        reranked.append(
            {
                **r,
                "lexical_overlap": lexical_overlap,
                "meta_overlap": meta_overlap,
                "combined_score": combined,
            }
        )

    # tri décroissant sur le score combiné
    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return reranked



# ============================
# RÉPONSE (RAG "classique")
# ============================

def answer_question(question: str, top_k: int = TOP_K) -> str:
    """
    Pipeline RAG :
      1) Retrieval dense (CamemBERT + FAISS)
      2) Rerank lexical (overlap mots question/document)
      3) On renvoie les champions les plus probables avec une phrase claire
    """

    raw_results = search_similar_docs(question, top_k)
    if not raw_results:
        return "Je ne trouve rien dans ma base de connaissances LoL pour cette question 😅"

    results = rerank_results(question, raw_results)

    # On décide : si l'utilisateur demande "plusieurs" ou "des champions", on en renvoie plusieurs
    q_lower = question.lower()
    wants_many = any(w in q_lower for w in ["plusieurs", "des champions", "quels", "quelles"])

    if wants_many:
        # On renvoie par exemple les 3 premiers
        top = results[:3]
        names = [r["doc"]["name"] for r in top]
        debug = ", ".join(
            f"{r['doc']['name']} (score={r['combined_score']:.2f}, overlap={r['lexical_overlap']})"
            for r in top
        )
        return (
            f"Les champions qui correspondent le mieux à ta question sont : {', '.join(names)}.\n\n"
            f"_Debug (pour le projet) : {debug}_"
        )
    else:
        # On renvoie le meilleur
        best = results[0]
        name = best["doc"]["name"]
        debug = f"score={best['combined_score']:.2f}, overlap={best['lexical_overlap']}, meta={best['meta_overlap']}, score_FAISS={best['score']:.2f}"
        return (
            f"Le champion qui correspond le mieux à ta question est : **{name}**.\n\n"
            f"_Debug (pour le projet) : {debug}_"
        )


# ============================
# UI STREAMLIT (CHATBOT)
# ============================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(page_title="LoL RAG Chatbot", page_icon="🧠")
    st.title("🧠💬 LoL RAG Chatbot (CamemBERT + FAISS + rerank lexical)")
    st.caption(
        "Pose des questions sur les champions de League of Legends. "
        "Le bot utilise un RAG : embeddings CamemBERT + FAISS, puis reranking lexical."
    )

    init_session_state()

    # Afficher l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Champ de saisie
    user_input = st.chat_input("Pose ta question (ex : Quel champion est un mage d'Ionia ?)")
    if user_input:
        # Ajout message user
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Affichage user
        with st.chat_message("user"):
            st.markdown(user_input)

        # Réponse du bot
        with st.chat_message("assistant"):
            with st.spinner("Je fouille dans le Grimoire de Runeterra..."):
                answer = answer_question(user_input)
                st.markdown(answer)

        # Ajout message assistant
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
