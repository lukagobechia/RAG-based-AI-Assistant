from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ask_question


app = FastAPI(
    title="RAG AI Assistant",
    description="Gemini-powered RAG AI Assistant",
    version="1.0.0"
)


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def root():

    return {
        "message": "RAG AI Assistant is running"
    }


@app.post("/ask")
def ask(request: QuestionRequest):

    return ask_question(
        request.question
    )