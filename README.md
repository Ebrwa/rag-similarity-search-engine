# RAG-Based Similarity Search Engine

A Retrieval-Augmented Generation (RAG) application that combines semantic search with Large Language Models (LLMs) to provide accurate, context-aware answers from retrieved documents.

The system processes web-based documents, converts them into vector embeddings, stores them in a FAISS vector database, retrieves the most relevant information using semantic similarity search, and uses an LLM to generate grounded responses with source references.

## Features

- Web document ingestion from source URLs.
- Automatic text extraction using BeautifulSoup.
- Document chunking for efficient retrieval.
- Generate embeddings using HuggingFace models.
- Store and search embeddings using FAISS vector database.
- Semantic similarity search to retrieve relevant context.
- Retrieval-Augmented Generation (RAG) pipeline.
- Integration with Groq API and Llama 3.3 70B.
- Generate context-aware answers based on retrieved documents.
- Display original source titles and URLs.

## Architecture

```
User Question
      |
      ↓
Query Processing
      |
      ↓
FAISS Vector Database
      |
      ↓
Relevant Document Retrieval
      |
      ↓
Context Construction
      |
      ↓
LLM Generation (Llama 3.3)
      |
      ↓
Final Grounded Answer
```

## Technologies Used

- Python
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- Groq API
- Llama 3.3 70B Versatile
- BeautifulSoup
- Requests

## Project Structure

```
rag-similarity-search-engine/
│
├── main.py
├── human_rights_links-2.csv
├── requirements.txt
├── .gitignore
└── .env (not included)
```

## How It Works

1. Load document URLs from a CSV file.
2. Extract webpage content using BeautifulSoup.
3. Split documents into smaller chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in a FAISS vector database.
6. Receive a user question.
7. Retrieve the most relevant document chunks.
8. Send the retrieved context with the question to the LLM.
9. Generate a grounded answer with source references.

## Installation

Clone the repository:

```bash
git clone https://github.com/Ebrwa/similarity-search-engine.git
cd similarity-search-engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
python main.py
```

## Example

**Question:**

```
What is the right to education?
```

**Workflow:**

```
Retrieve relevant documents
          ↓
Create context
          ↓
Send context to Llama 3.3
          ↓
Generate grounded response
          ↓
Display answer with sources
```

## Future Improvements

- Add conversational memory for multi-turn conversations.
- Implement advanced retrieval techniques such as MMR and Hybrid Search.
- Add support for PDF and multiple document formats.
- Build a web-based user interface.
- Support multiple LLM providers.

## Author

**Ibrahim Blih**

Software Engineering Student interested in AI Engineering, LLM applications, and building practical software systems.
