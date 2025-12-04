import os
import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import unicodedata
import numpy as np
import torch
from transformers import CamembertModel, CamembertTokenizerFast
import faiss
import re
import functools

# ==============
# CONFIG GLOBALE
# ==============
STORAGE_BASE_URL = os.getenv("STORAGE_BASE_URL")  
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "champions"  

EMBED_MODEL_NAME = "camembert-base"
MAX_LENGTH = 256
TOP_K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "et", "en",
    "au", "aux", "pour", "avec", "que", "qui", "quel", "quelle",
    "quels", "quelles", "est", "sont", "champion", "champions", "donne",
    "moi", "je", "un", "une", "des"
}


# ============================
# CHARGEMENT DU MODELE
# ============================

@functools.lru_cache()
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


def _get_champion_filenames_from_storage() -> List[str]:
    """
    Récupère la liste des fichiers JSON depuis le pod de storage
    via un fichier index.json situé dans le dossier champions.

    STORAGE_BASE_URL doit être du type :
        http://rag-storage-service:9000/champions
    et on lit :
        {STORAGE_BASE_URL}/index.json
    """
    if not STORAGE_BASE_URL:
        raise RuntimeError("STORAGE_BASE_URL n'est pas défini dans les variables d'environnement.")

    base = STORAGE_BASE_URL.rstrip("/")
    index_url = f"{base}/index.json"

    resp = requests.get(index_url, timeout=10)
    resp.raise_for_status()
    filenames = resp.json()

    if not isinstance(filenames, list) or not filenames:
        raise RuntimeError(f"index.json mal formé ou vide à l'URL {index_url}")

    return sorted(filenames)


def _load_champion_from_storage(filename: str) -> Dict[str, Any]:
    """
    Charge un champion individuel depuis le pod de storage via HTTP.
    """
    base = STORAGE_BASE_URL.rstrip("/")
    url = f"{base}/{filename}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def load_champion_docs() -> List[Dict[str, Any]]:
    """
    Charge TOUS les champions UNIQUEMENT depuis le pod de storage (HTTP),
    à partir de STORAGE_BASE_URL et d'un index.json.
    Aucun accès disque local, conformément à la contrainte du projet.
    """
    filenames = _get_champion_filenames_from_storage()
    docs: List[Dict[str, Any]] = []

    for fname in filenames:
        champ = _load_champion_from_storage(fname)
        text = build_text_from_champion(champ)
        docs.append(
            {
                "id": fname.rsplit(".", 1)[0],
                "name": champ.get("name", fname),
                "raw": champ,
                "text": text,
            }
        )

    if not docs:
        raise RuntimeError("Aucun champion chargé depuis le pod de storage (liste vide).")

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
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        return cls_embeddings.cpu().numpy()


# ============================
# INDEX FAISS (RETRIEVAL)
# ============================

@functools.lru_cache()
def build_index():
    tokenizer, model = load_camembert()
    docs = load_champion_docs()

    if not docs:
        raise RuntimeError("Aucun champion trouvé (storage + disque).")

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


# ============================
# RERANK LEXICAL
# ============================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def simple_tokenize(text: str) -> List[str]:
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


def rerank_results(question: str, results: List[Dict[str, Any]], alpha=1.0, beta=0.3, gamma=0.8):
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
# FONCTION FINALE
# ============================

def answer_question(question: str, top_k: int = TOP_K) -> str:
    raw_results = search_similar_docs(question, top_k)
    if not raw_results:
        return "Je ne trouve rien dans ma base de connaissances LoL pour cette question 😅"

    results = rerank_results(question, raw_results)

    q_lower = question.lower()
    wants_many = any(w in q_lower for w in ["plusieurs", "des champions", "quels", "quelles"])

    if wants_many:
        top = results[:3]
        names = [r["doc"]["name"] for r in top]
        debug = ", ".join(
            f"{r['doc']['name']} (score={r['combined_score']:.2f}, overlap={r['lexical_overlap']})"
            for r in top
        )
        return (
            f"Les champions qui correspondent le mieux à ta question sont : {', '.join(names)}.\n\n"
            f"_Debug : {debug}_"
        )
    else:
        best = results[0]
        name = best["doc"]["name"]
        debug = (
            f"score={best['combined_score']:.2f}, "
            f"overlap={best['lexical_overlap']}, "
            f"meta={best['meta_overlap']}, "
            f"score_FAISS={best['score']:.2f}"
        )
        return (
            f"Le champion qui correspond le mieux à ta question est : **{name}**.\n\n"
            f"_Debug : {debug}_"
        )
