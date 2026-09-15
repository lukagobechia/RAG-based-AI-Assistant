import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def ingest_documents():

    # loading documents

    documents = []

    # txt
    documents.extend(
        DirectoryLoader(
            str(DOCUMENTS_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader
        ).load()
    )

    # pdf
    documents.extend(
        DirectoryLoader(
            str(DOCUMENTS_DIR),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        ).load()
    )

    print(f"Loaded {len(documents)} documents")


    # spliting documents into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")


    # createing Gemini embeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        output_dimensionality=1536
    )


    # storing embeddings in db

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print("Documents successfully indexed.")


if __name__ == "__main__":
    ingest_documents()