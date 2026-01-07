"""
Agent SQL utilisant Groq pour convertir les questions en requêtes SQL.
"""
import os
import sys
from pathlib import Path

# Ajoute le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from groq import Groq
from dotenv import load_dotenv
from src.database import get_schema, execute_query

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def question_to_sql(question: str) -> str:
    """Convertit une question en requête SQL via Groq."""
    
    schema = get_schema()
    
    prompt = f"""Tu es un expert SQL. Convertis la question utilisateur en requête SQLite.

{schema}

Règles:
- Retourne UNIQUEMENT la requête SQL, rien d'autre
- Utilise des JOINs pour les tables de relations
- Les comparaisons de texte sont case-insensitive (utilise LOWER() si besoin)
- Si la question n'est pas liée aux données, retourne: SELECT 'Question non pertinente' as error

Question: {question}

SQL:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )
    
    sql = response.choices[0].message.content.strip()
    # Nettoie le SQL (enlève les backticks markdown si présents)
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def format_results(question: str, results: dict) -> str:
    """Formate les résultats SQL en réponse naturelle via Groq."""
    
    if "error" in results:
        return f"Erreur SQL: {results['error']}"
    
    if not results.get("rows"):
        return "Aucun résultat trouvé pour cette requête."
    
    # Formate les résultats
    columns = results["columns"]
    rows = results["rows"]
    
    # Si peu de résultats, on peut les lister
    if len(rows) <= 10:
        formatted = "\n".join([str(row) for row in rows])
    else:
        formatted = f"{len(rows)} résultats. Premiers: {rows[:5]}"
    
    prompt = f"""Réponds à la question de l'utilisateur en français, de manière naturelle et concise.

Question: {question}
Données trouvées (colonnes: {columns}):
{formatted}

Réponse:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()


def ask_sql(question: str) -> str:
    """Pipeline complet: question -> SQL -> résultats -> réponse."""
    
    print(f"[SQL Agent] Question: {question}")
    
    # 1. Génère le SQL
    sql = question_to_sql(question)
    print(f"[SQL Agent] SQL: {sql}")
    
    # 2. Exécute
    results = execute_query(sql)
    print(f"[SQL Agent] Résultats: {len(results.get('rows', []))} lignes")
    
    # 3. Formate la réponse
    answer = format_results(question, results)
    
    return answer


if __name__ == "__main__":
    # Tests
    questions = [
        "Quels champions viennent d'Ionia ?",
        "Liste les mages",
        "Combien de champions jouent mid ?",
        "Liste les Tireur",
        "Quels ADC sont mobiles ?",
        "Comment fonctionne l'ultime de Kai'Sa" ,
        "Quel est le lien entre Jinx et Vi ?",
        "Raconte l'histoire de Akali",
        "Quel champion est facile pour débuter ?",
    ]
    
    for q in questions:
        print(f"\n{'='*50}")
        print(f"Q: {q}")
        print(f"R: {ask_sql(q)}")
