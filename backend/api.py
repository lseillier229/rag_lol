# backend/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import answer_question

app = FastAPI(title="LoL RAG Backend")

class Question(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/answer")
def get_answer(payload: Question):
    answer = answer_question(payload.question)
    return {"answer": answer}
