import os
import time

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME"
)

RELEVANCE_THRESHOLD = float(
    os.getenv("RELEVANCE_THRESHOLD", "0.72")
)


# gemini embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=1536
)


# pinecone

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)


# gemini llm

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# prompt engineering

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the
information provided in the context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer cannot be found in the context,
   say that you do not have enough information.
4. Give a concise and clear answer.
5. Base factual claims on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
)


# rag pipeline

def ask_question(question: str):

    # retrieve relevant documents
    start = time.time()

    results = vector_store.similarity_search_with_score(
        question,
        k=4
    )

    retrieval_time = time.time() - start


    # filter results by relevance
    filtered_results = [
        (document, score)
        for document, score in results
        if score >= RELEVANCE_THRESHOLD
    ]


    if not filtered_results:
        return {
            "answer": "I don't have enough information to answer this question.",
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0
        }


    # extracting documents
    documents = [
        document
        for document, score in filtered_results
    ]


    # building context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # building prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )


    # generating answer with Gemini
    start = time.time()

    response = llm.invoke(messages)

    generation_time = time.time() - start


    # extract sources
    sources = []

    for document, score in filtered_results:
        sources.append({
            "source": document.metadata.get("source"),
            "content": document.page_content,
            "score": score
        })


    # debug
    for document, score in results:
        print("=" * 50)
        print("Score:", score)
        print("Source:", document.metadata.get("source"))
        print("Content:", document.page_content)
        print("Retrieval Time:", retrieval_time)
        print("Generation Time:", generation_time)
        print()


    return {
        "answer": response.content,
        "sources": sources,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time
    }