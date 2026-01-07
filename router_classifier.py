"""
Classifieur de routage SQL/RAG basé sur les embeddings.
Utilise un SVM entraîné sur des exemples pour classifier les questions.
"""
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

# Chemin du modèle sauvegardé
MODEL_PATH = Path("data/router_model.pkl")

# Dataset d'entraînement
TRAINING_DATA = [
    # === SQL : questions factuelles, listes, comptages, compétences ===
    ("Quels champions viennent d'Ionia ?", "SQL"),
    ("Liste les mages", "SQL"),
    ("Combien de champions jouent mid ?", "SQL"),
    ("Quels ADC sont dans la base ?", "SQL"),
    ("Quels assassins jouent jungle ?", "SQL"),
    ("Liste les supports", "SQL"),
    ("Quels champions viennent de Noxus ?", "SQL"),
    ("Quelles sont les compétences de Kai'Sa ?", "SQL"),
    ("Décris le Q de Jinx", "SQL"),
    ("Quel est l'ultime d'Irelia ?", "SQL"),
    ("Quels sont les pouvoirs de Kai'Sa ?", "SQL"),
    ("Quelles sont les capacités de Zed ?", "SQL"),
    ("Quelles sont les compétences d'Akali ?", "SQL"),
    ("Quel est le passif de Yasuo ?", "SQL"),
    ("Combien d'assassins dans la base ?", "SQL"),
    ("Liste les champions de Demacia", "SQL"),
    ("Quels tanks jouent top ?", "SQL"),
    ("Quel est le W de Thresh ?", "SQL"),
    ("Quels champions ont un dash ?", "SQL"),
    ("Liste les junglers", "SQL"),
    ("Quels mages viennent d'Ionia ?", "SQL"),
    ("Combien de supports dans la base ?", "SQL"),
    ("Quel est le E de Lee Sin ?", "SQL"),
    ("Quels champions jouent bot ?", "SQL"),
    ("Liste les combattants", "SQL"),
    ("Quel est l'ultime de Lux ?", "SQL"),
    ("Quels champions de Piltover ?", "SQL"),
    ("Décris le R de Jinx", "SQL"),
    ("Quelles lanes pour Ahri ?", "SQL"),
    ("Quel rôle joue Thresh ?", "SQL"),
    
    # === RAG : lore, histoire, background, conseils, relations ===
    ("Raconte l'histoire d'Akali", "RAG"),
    ("Quel est le lore de Yasuo ?", "RAG"),
    ("Comment Kai'Sa a-t-elle survécu dans le Void ?", "RAG"),
    ("Quelle est l'histoire d'Ahri ?", "RAG"),
    ("Raconte le lore de Thresh", "RAG"),
    ("Qui est Zed ?", "RAG"),
    ("Explique le background de Jinx", "RAG"),
    ("Quel est le lien entre Jinx et Vi ?", "RAG"),
    ("Quel est le background de Lee Sin ?", "RAG"),
    ("Qui est le père de Kai'Sa ?", "RAG"),
    ("Comment Ahri utilise-t-elle ses pouvoirs ?", "RAG"),
    ("Comment Thresh capture-t-il les âmes ?", "RAG"),
    ("Donne des conseils pour jouer Akali", "RAG"),
    ("Comment bien jouer Yasuo ?", "RAG"),
    ("Quelle est l'origine de Viego ?", "RAG"),
    ("Pourquoi Yasuo est-il exilé ?", "RAG"),
    ("Raconte le passé de Riven", "RAG"),
    ("Quelle est la relation entre Zed et Shen ?", "RAG"),
    ("Comment Jhin est-il devenu tueur ?", "RAG"),
    ("Explique l'histoire des Îles Obscures", "RAG"),
    ("Qui a créé Blitzcrank ?", "RAG"),
    ("Pourquoi Thresh est-il devenu spectre ?", "RAG"),
    ("Quel est le passé de Katarina ?", "RAG"),
    ("Comment jouer Lee Sin en jungle ?", "RAG"),
    ("Tips pour Zed mid ?", "RAG"),
    ("Stratégie pour Jinx en teamfight ?", "RAG"),
    ("Comment utiliser l'ultime de Yasuo ?", "RAG"),
    ("Quel est le lore de Morgana et Kayle ?", "RAG"),
    ("Raconte la guerre entre Ionia et Noxus", "RAG"),
    ("Qui sont les Kinkou ?", "RAG"),
]


def train_router():
    """Entraîne le classifieur de routage."""
    print("Chargement du modèle d'embeddings...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Prépare les données
    questions = [q for q, _ in TRAINING_DATA]
    labels = [1 if label == "SQL" else 0 for _, label in TRAINING_DATA]
    
    print(f"Encodage de {len(questions)} exemples...")
    embeddings = model.encode(questions, show_progress_bar=True)
    
    # Entraîne le SVM
    print("Entraînement du classifieur SVM...")
    classifier = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
    classifier.fit(embeddings, labels)
    
    # Validation croisée
    scores = cross_val_score(classifier, embeddings, labels, cv=5)
    print(f"Accuracy (5-fold CV): {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
    
    # Sauvegarde
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    return classifier


def load_router():
    """Charge le classifieur de routage."""
    if not MODEL_PATH.exists():
        print("Modèle non trouvé, entraînement...")
        return train_router()
    
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


class RouterClassifier:
    """Classifieur de routage SQL/RAG."""
    
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.classifier = load_router()
    
    def predict(self, question: str) -> str:
        """Prédit si la question est SQL ou RAG."""
        embedding = self.model.encode([question])
        prediction = self.classifier.predict(embedding)[0]
        return "SQL" if prediction == 1 else "RAG"
    
    def predict_proba(self, question: str) -> dict:
        """Retourne les probabilités pour chaque classe."""
        embedding = self.model.encode([question])
        proba = self.classifier.predict_proba(embedding)[0]
        return {"RAG": proba[0], "SQL": proba[1]}


if __name__ == "__main__":
    # Entraîne le modèle
    train_router()
    
    # Test
    router = RouterClassifier()
    
    test_questions = [
        "Quel est le background de Lee Sin ?",
        "Quelles sont les compétences de Zed ?",
        "Raconte l'histoire de Jinx",
        "Liste les assassins",
        "Comment jouer Akali ?",
        "Quel est l'ultime de Thresh ?",
    ]
    
    print("\n=== Tests ===")
    for q in test_questions:
        pred = router.predict(q)
        proba = router.predict_proba(q)
        print(f"[{pred}] ({proba[pred]:.1%}) {q}")
