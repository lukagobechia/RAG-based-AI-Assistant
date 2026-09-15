# RAG AI Assistant

A Retrieval-Augmented Generation (RAG) AI assistant built with **Python, LangChain, Google Gemini, Pinecone, and FastAPI**.

The assistant retrieves relevant information from a document knowledge base and generates answers based only on the retrieved context.

## Tech Stack

* Python
* LangChain
* Google Gemini
* Gemini Embeddings
* Pinecone Vector Database
* FastAPI
* Pydantic

## Architecture

```text
Documents
   ↓
Chunking
   ↓
Gemini Embeddings
   ↓
Pinecone
   ↓
User Question
   ↓
Semantic Search
   ↓
Relevant Context
   ↓
Gemini
   ↓
Answer + Sources
```

## Features

* PDF and TXT document ingestion
* Document chunking and embeddings
* Semantic vector search
* Relevance filtering
* RAG-based answer generation
* Source attribution
* Hallucination prevention
* REST API with FastAPI

## Setup

```bash
git clone <repository-url>
cd rag-ai-assistant

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-assistant
RELEVANCE_THRESHOLD=0.72
```

Create a Pinecone index with **1536 dimensions**, then ingest the documents:

```bash
python app/ingest.py
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8005
```

API documentation:

```text
http://127.0.0.1:8005/docs
```

## API

### `POST /ask`

```json
{
  "question": "How many users does the Basic plan support?"
}
```

Example:

```json
{
  "answer": "The Basic plan supports up to 5 users.",
  "sources": [
    {
      "source": "documents/company.txt",
      "score": 0.8146
    }
  ]
}
```

## Evaluation

The system was tested with **11 questions**:

* **6/6** answerable questions answered correctly
* **5/5** unanswerable or irrelevant questions correctly refused
* **11/11** overall correct

The assistant uses retrieved context and a relevance threshold to reduce unsupported answers and hallucinations.

## Project Structure

```text
rag-ai-assistant/
├── app/
│   ├── main.py
│   ├── rag.py
│   └── ingest.py
├── documents/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## License

Educational and portfolio project.
# RAG-based-AI-Assistant
