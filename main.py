"""
LoL RAG Chatbot - Version hybride
- Routeur ML (SVM + embeddings) pour choisir le pipeline
- Agent SQL pour les questions factuelles
- RAG (Sentence-Transformers multilingue + FAISS + BM25 + Reranker) pour les questions ouvertes sur le wiki
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import streamlit as st
import faiss
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from src.database import get_schema, execute_query
from src.router_classifier import RouterClassifier

load_dotenv()

# ==============
# CONFIG GLOBALE
# ==============

WIKI_DIR = Path("data/unstructured")  # Fichiers wiki scrapés
# Modèle multilingue pour gérer FR questions + EN documents
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cross-encoder multilingue pour le reranking
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
TOP_K = 15  # Récupère plus de candidats pour le reranking
TOP_K_RERANK = 10  # Garde les 10 meilleurs après reranking

# Client Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================
# CHARGEMENT DES MODELES
# ============================

@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embedding multilingue."""
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return model


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

def load_reranker():
    """Charge le cross-encoder pour le reranking."""
    return CrossEncoder(RERANKER_MODEL_NAME)



# ============================
# CHARGEMENT DES DOCUMENTS WIKI
# ============================

def load_wiki_docs() -> List[Dict[str, Any]]:
    """Charge tous les fichiers wiki du dossier data/unstructured."""
    docs = []
    
    if not WIKI_DIR.exists():
        return docs
    
    for path in sorted(WIKI_DIR.glob("*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        name = path.stem
        
        docs.append({
            "id": name,
            "name": name,
            "text": content,
            "source": str(path)
        })
    
    return docs


def chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Découpe un document en chunks par sections (##) avec overlap."""
    chunks = []
    
    # Essaie de découper par sections markdown
    sections = text.split("\n## ")
    
    if len(sections) > 1:
        # Premier chunk (avant le premier ##)
        if sections[0].strip():
            first_chunk = sections[0][:chunk_size]
            chunks.append(first_chunk)
        
        # Autres sections
        for section in sections[1:]:
            section_text = "## " + section
            if len(section_text) > chunk_size:
                # Si section trop longue, découpe en sous-chunks avec overlap
                start = 0
                while start < len(section_text):
                    end = start + chunk_size
                    chunks.append(section_text[start:end])
                    start = end - overlap  # Overlap avec le chunk précédent
            else:
                chunks.append(section_text)
    else:
        # Pas de sections, découpe par taille avec overlap
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
    
    return chunks


def is_useful_chunk(text: str) -> bool:
    """Vérifie si un chunk contient du contenu utile (pas juste des liens)."""
    # Ignore les chunks trop courts
    if len(text) < 100:
        return False
    
    # Ignore les chunks qui sont principalement des liens/références
    useless_patterns = ["## See also", "## References", "## Changelog", "## Patch history", 
                        "Category:", "(TBA)", "## Champion skins"]
    for pattern in useless_patterns:
        if pattern in text and len(text) < 300:
            return False
    
    return True


def enrich_chunk_text(text: str, champion_name: str) -> str:
    """Enrichit le texte des chunks pour améliorer le retrieval.
    
    Ajoute des mots-clés FR/EN aux sections conseils pour mieux matcher
    les questions comme 'comment jouer X' vs 'comment contrer X'.
    """
    # Enrichit les conseils alliés (pour jouer)
    text = text.replace(
        "## Conseils (alliés)",
        f"## Conseils pour jouer {champion_name} (alliés) - Tips to play {champion_name}"
    )
    # Enrichit les conseils contre (pour contrer)
    text = text.replace(
        "## Conseils (contre)",
        f"## Conseils pour contrer {champion_name} (contre) - Tips against {champion_name}"
    )
    # Enrichit le lore/background
    text = text.replace(
        "## Lore",
        f"## Lore - Histoire et passé de {champion_name} - Background story history"
    )
    text = text.replace(
        "## Background",
        f"## Background - Histoire et passé de {champion_name} - Lore story history"
    )
    return text


def load_wiki_chunks() -> List[Dict[str, Any]]:
    """Charge les documents wiki et les découpe en chunks avec préfixe champion."""
    chunks = []
    
    if not WIKI_DIR.exists():
        return chunks
    
    for path in sorted(WIKI_DIR.glob("*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        name = path.stem
        # Nom du champion formaté (première lettre majuscule)
        champion_name = name.capitalize()
        doc_chunks = chunk_document(content)
        
        for i, chunk_text in enumerate(doc_chunks):
            # Filtre les chunks inutiles
            if not is_useful_chunk(chunk_text):
                continue
            
            # Enrichit le texte pour améliorer le retrieval
            enriched_text = enrich_chunk_text(chunk_text, champion_name)
            
            # Ajoute le nom du champion en préfixe pour le contexte
            prefixed_text = f"[Champion: {champion_name}]\n{enriched_text}"
                
            chunks.append({
                "id": f"{name}_{i}",
                "name": name,
                "text": prefixed_text,
                "source": str(path),
                "chunk_idx": i
            })
    
    return chunks


# ============================
# EMBEDDINGS (MULTILINGUE)
# ============================

def embed_texts(texts: List[str], model: SentenceTransformer, is_query: bool = False) -> np.ndarray:
    """Calcule les embeddings avec le modèle multilingue."""
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings).astype('float32')


# ============================
# INDEX FAISS + BM25 (avec cache disque)
# ============================

INDEX_CACHE_PATH = Path("data/wiki_index.pkl")

def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25."""
    # Simple tokenization: lowercase + split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def save_index(chunks, embeddings, bm25, docs):
    """Sauvegarde l'index sur disque."""
    import pickle
    with open(INDEX_CACHE_PATH, 'wb') as f:
        pickle.dump({
            "chunks": chunks,
            "embeddings": embeddings,
            "bm25": bm25,
            "docs": docs
        }, f)


def load_index_from_cache():
    """Charge l'index depuis le disque."""
    import pickle
    if INDEX_CACHE_PATH.exists():
        with open(INDEX_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_resource
def build_wiki_index():
    """Construit ou charge l'index FAISS + BM25 sur les chunks de documents wiki."""
    model = load_embedding_model()
    
    # Essaie de charger depuis le cache
    cached = load_index_from_cache()
    if cached:
        print("Index chargé depuis le cache disque")
        chunks = cached["chunks"]
        embeddings = cached["embeddings"]
        bm25 = cached["bm25"]
        docs = cached["docs"]
        
        # Reconstruit l'index FAISS (rapide)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        return {
            "model": model,
            "chunks": chunks,
            "docs": docs,
            "index": index,
            "bm25": bm25,
        }
    
    # Sinon, construit l'index
    print("Construction de l'index wiki...")
    chunks = load_wiki_chunks()
    docs = load_wiki_docs()

    if not chunks:
        st.warning(f"Aucun document wiki trouvé dans {WIKI_DIR}")
        return None

    # Embedding de chaque chunk (FAISS)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # BM25 index
    tokenized_chunks = [tokenize_text(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Sauvegarde pour la prochaine fois
    save_index(chunks, embeddings, bm25, docs)
    print("Index sauvegardé sur disque")

    return {
        "model": model,
        "chunks": chunks,
        "docs": docs,
        "index": index,
        "bm25": bm25,
    }


# ============================
# ROUTEUR ML (SVM + Embeddings)
# ============================

# Charge le classifieur au démarrage
@st.cache_resource
def load_router():
    """Charge le classifieur de routage."""
    return RouterClassifier()


def route_question(question: str) -> str:
    """Détermine si la question est factuelle (SQL) ou ouverte (RAG)."""
    router = load_router()
    return router.predict(question)


# ============================
# AGENT SQL
# ============================

def ask_sql(question: str) -> str:
    """Pipeline SQL : question -> requête -> résultat -> réponse."""
    
    schema = get_schema()
    
    prompt_sql = f"""Tu es un expert SQL. Convertis la question en requête SQLite.

{schema}

Règles:
- Retourne UNIQUEMENT la requête SQL, rien d'autre
- Utilise des JOINs pour les tables de relations
- Pour les noms de champions, utilise l'ID en minuscule sans apostrophe: jinx, kaisa, khazix, leesin
- Pour les abilities d'un champion: SELECT * FROM abilities WHERE champion_id = 'nom_champion' AND key = 'Q/W/E/R'
- Pour filtrer par rôle ET lane, utilise deux JOINs séparés

Exemples:
- "Décris le Q de Jinx" → SELECT name, description FROM abilities WHERE champion_id = 'jinx' AND key = 'Q'
- "Quel est l'ultime de Zed" → SELECT name, description FROM abilities WHERE champion_id = 'zed' AND key = 'R'
- "Quels champions viennent de Noxus" → SELECT name FROM champions WHERE region = 'Noxus'
- "Liste les mages" → SELECT c.name FROM champions c JOIN champion_roles r ON c.id = r.champion_id WHERE r.role = 'Mage'
- "Quels assassins jouent jungle" → SELECT c.name FROM champions c JOIN champion_roles r ON c.id = r.champion_id JOIN champion_lanes l ON c.id = l.champion_id WHERE r.role = 'Assassin' AND l.lane = 'jungle'

Question: {question}

SQL:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt_sql}],
        temperature=0,
        max_tokens=500
    )
    
    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    results = execute_query(sql)
    
    if "error" in results:
        return f"Erreur SQL: {results['error']}"
    
    if not results.get("rows"):
        return "Aucun résultat trouvé."
    
    rows = results["rows"]
    formatted = "\n".join([str(row) for row in rows[:15]])
    
    prompt_answer = f"""Réponds à la question en français, de manière naturelle et concise.

Question: {question}
Données: {formatted}

Réponse:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt_answer}],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()


# ============================
# RAG (WIKI) - MULTILINGUE
# ============================

def find_champion_in_question(question: str, docs: list) -> str:
    """Détecte si un nom de champion est mentionné dans la question."""
    question_lower = question.lower()
    # Normalise la question (enlève espaces et apostrophes pour matcher)
    question_normalized = question_lower.replace("'", "").replace(" ", "").replace("-", "")
    
    for doc in docs:
        name = doc["name"].lower()
        name_clean = name.replace("'", "").replace(" ", "").replace("-", "")
        # Match soit le nom exact, soit le nom normalisé
        if name in question_lower or name_clean in question_normalized:
            return doc["name"]
    return None


def expand_query(question: str) -> str:
    """Traduit la question en anglais pour améliorer le retrieval."""
    prompt = f"""Translate this French question to English. Return ONLY the English translation, nothing else.

French: {question}
English:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        translated = response.choices[0].message.content.strip()
        # Combine FR + EN
        return f"{question} {translated}"
    except Exception:
        return question


def generate_multi_queries(question: str) -> List[str]:
    """Génère plusieurs variantes de la question pour améliorer le retrieval."""
    prompt = f"""Tu es un expert en recherche d'information. Génère 3 reformulations de cette question pour améliorer la recherche dans une base de documents sur League of Legends.

Règles:
- 1 variante en français (reformulation différente)
- 2 variantes en anglais (traduction + reformulation avec termes techniques LoL)
- Utilise des synonymes et termes spécifiques au jeu
- Retourne UNIQUEMENT les 3 variantes, une par ligne, sans numérotation

Question originale: {question}

Variantes:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Modèle rapide pour cette tâche simple
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        variants = response.choices[0].message.content.strip().split("\n")
        # Nettoie et filtre les lignes vides
        variants = [v.strip().lstrip("0123456789.-) ") for v in variants if v.strip()]
        # Ajoute la question originale
        return [question] + variants[:3]
    except Exception:
        return [question]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion (RRF) pour combiner plusieurs listes de résultats.
    
    RRF score = sum(1 / (k + rank)) pour chaque liste où le document apparaît.
    k=60 est la valeur standard qui fonctionne bien en pratique.
    
    Args:
        rankings: Liste de listes d'indices de chunks (ordonnés par pertinence)
        k: Paramètre de lissage (défaut: 60)
    
    Returns:
        Dict mapping chunk_idx -> RRF score
    """
    rrf_scores = {}
    
    for ranking in rankings:
        for rank, chunk_idx in enumerate(ranking):
            if chunk_idx not in rrf_scores:
                rrf_scores[chunk_idx] = 0.0
            # RRF formula: 1 / (k + rank + 1) car rank commence à 0
            rrf_scores[chunk_idx] += 1.0 / (k + rank + 1)
    
    return rrf_scores


def ask_rag(question: str) -> str:
    """Pipeline RAG hybride : FAISS (sémantique) + BM25 (mots-clés) + Reranker."""
    
    store = build_wiki_index()
    
    if store is None:
        return "Base de connaissances wiki non disponible."
    
    model = store["model"]
    chunks = store["chunks"]
    docs = store["docs"]
    index = store["index"]
    bm25 = store["bm25"]
    
    # Charge le reranker
    reranker = load_reranker()
    
    # 1. Détecte si un champion est mentionné
    champion_name = find_champion_in_question(question, docs)
    
    # 2. Recherche hybride : FAISS + BM25
    query_emb = embed_texts([question], model, is_query=True)
    k = min(TOP_K, len(chunks))
    faiss_scores, faiss_indices = index.search(query_emb, k)
    
    # BM25 (mots-clés)
    query_tokens = tokenize_text(question)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
    
    # 3. Combine les résultats (fusion des scores) - FAISS a plus de poids
    chunk_scores = {}
    
    # FAISS score (poids x2 car plus fiable pour la sémantique)
    for score, idx in zip(faiss_scores[0], faiss_indices[0]):
        chunk_scores[int(idx)] = chunk_scores.get(int(idx), 0) + score * 2
    
    # BM25 score (poids x1)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    for idx in bm25_top_indices:
        normalized_score = bm25_scores[idx] / max_bm25
        chunk_scores[int(idx)] = chunk_scores.get(int(idx), 0) + normalized_score
    
    # 3b. BOOST: Si un champion est mentionné, ajoute TOUS ses chunks au pool de candidats
    # Cela garantit que les chunks pertinents du champion sont considérés pour le reranking
    champion_chunk_indices = []
    if champion_name:
        for i, chunk in enumerate(chunks):
            if chunk["name"].lower() == champion_name.lower():
                champion_chunk_indices.append(i)
                if i not in chunk_scores:
                    chunk_scores[i] = 0.5
    
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Reranking avec cross-encoder
    # Prend les top K + tous les chunks du champion mentionné
    candidate_indices = [idx for idx, _ in sorted_chunks[:TOP_K]]
    # Ajoute les chunks du champion qui ne sont pas déjà dans les candidats
    for idx in champion_chunk_indices:
        if idx not in candidate_indices:
            candidate_indices.append(idx)
    pairs = [(question, chunks[idx]["text"]) for idx in candidate_indices]
    
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
    
    # 5. Construit le contexte avec les meilleurs chunks après reranking
    context_parts = []
    seen_chunks = set()
    
    # Priorise les chunks du champion mentionné parmi les top reranked
    if champion_name:
        for idx, score in reranked:
            chunk = chunks[idx]
            if chunk["name"].lower() == champion_name.lower():
                chunk_id = chunk["id"]
                if chunk_id not in seen_chunks:
                    context_parts.append(f"=== {chunk['name']} (rerank score: {score:.2f}) ===\n{chunk['text']}")
                    seen_chunks.add(chunk_id)
                    if len(seen_chunks) >= TOP_K_RERANK:
                        break
    
    # Ajoute les autres chunks reranked
    for idx, score in reranked:
        chunk = chunks[idx]
        chunk_id = chunk["id"]
        if chunk_id not in seen_chunks:
            context_parts.append(f"=== {chunk['name']} (rerank score: {score:.2f}) ===\n{chunk['text']}")
            seen_chunks.add(chunk_id)
        
        # Limite au top K après reranking
        if len(seen_chunks) >= TOP_K_RERANK:
            break
    
    context = "\n\n".join(context_parts)
    
    # 6. Génération avec Groq
    prompt = f"""Tu es un expert de League of Legends. Réponds à la question en FRANÇAIS en utilisant UNIQUEMENT le contexte fourni.

RÈGLES IMPORTANTES:
- Lis ATTENTIVEMENT TOUT le contexte avant de répondre
- Cherche les informations pertinentes dans TOUTES les sections, pas seulement les premières
- Pour les questions sur les pouvoirs/capacités, cherche les sections "Abilities" ou "Soul Snatching"
- Utilise SEULEMENT les informations présentes dans le contexte
- Si l'information demandée n'est PAS dans le contexte, réponds: "Cette information n'est pas disponible dans ma base de connaissances."
- Ne jamais inventer ou deviner des informations
- Le contexte peut être en anglais, traduis en français

Contexte:
{context}

Question: {question}

Réponse (en français, basée uniquement sur le contexte):"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()


# ============================
# PIPELINE PRINCIPAL
# ============================

def answer_question(question: str) -> tuple[str, str]:
    """Pipeline complet: route + SQL/RAG."""
    
    route = route_question(question)
    
    if route == "SQL":
        answer = ask_sql(question)
    else:
        answer = ask_rag(question)
    
    return answer, route


# ============================
# UI STREAMLIT
# ============================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(page_title="LoL RAG Chatbot", page_icon="🎮")
    st.title("🎮 LoL Chatbot Hybride")
    st.caption(
        "Pose des questions sur League of Legends. "
        "SQL pour les questions factuelles, RAG multilingue pour les questions ouvertes."
    )

    init_session_state()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Pose ta question...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                answer, route = answer_question(user_input)
                
                route_emoji = "🗃️ SQL" if route == "SQL" else "📚 RAG"
                st.caption(f"Pipeline: {route_emoji}")
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
