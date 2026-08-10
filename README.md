# Conversational RAG Application

This project is a Retrieval-Augmented Generation (RAG) application for answering questions about human rights documents.

The application retrieves relevant information from a FAISS vector database and uses a Groq language model to generate answers based only on the retrieved sources.

The project was improved to support conversational memory and advanced retrieval techniques.

## Features

- Web content extraction using BeautifulSoup
- Text splitting using RecursiveCharacterTextSplitter
- Hugging Face embeddings
- FAISS vector database
- Groq LLM for answer generation
- Conversational memory
- Follow-up question understanding
- Basic Similarity Search
- Maximum Marginal Relevance (MMR)
- Multi-Query Retrieval
- Retrieval performance comparison
- Source display for generated answers

## Conversational Memory

The chatbot stores previous user questions and assistant answers during the conversation.

When the user asks a follow-up question, the application uses the conversation history to rewrite it as a standalone question before performing retrieval.

Example:

First question:

What is the right to education?

Follow-up question:

Who should protect it?

The application can rewrite the follow-up question as:

Who is responsible for protecting the right to education?

This allows the chatbot to maintain context without requiring the user to repeat previous information.

## Retrieval Methods

The application compares three retrieval methods.

### 1. Basic Similarity Search

Basic similarity search retrieves the document chunks that are most semantically similar to the user's question.

This method is used as the baseline for comparison.

### 2. Maximum Marginal Relevance (MMR)

MMR retrieves relevant documents while also reducing redundant results.

It attempts to balance relevance and diversity between the retrieved document chunks.

### 3. Multi-Query Retrieval

Multi-Query Retrieval uses the language model to generate multiple versions of the user's question.

Each generated query is searched separately in the FAISS vector database. The retrieved results are then combined and duplicate chunks are removed.

This can improve retrieval coverage because the same information may be expressed using different wording.

## Retrieval Comparison

The application compares the retrieval methods using:

- Retrieval time
- Number of retrieved results
- Number of unique sources

Example test question:

What is the right to education?

Example results:

| Method | Results | Unique Sources |
|---|---:|---:|
| Basic Similarity Search | 3 | 2 |
| MMR | 3 | 3 |
| Multi-Query Retrieval | 5 | 4 |

In this test, Basic Similarity Search was the fastest approach.

MMR returned more diverse sources than basic similarity search.

Multi-Query Retrieval required more time because it generated and searched multiple queries, but it retrieved information from a larger number of unique sources.

## RAG Pipeline

The application follows this pipeline:

1. Read URLs from the CSV file.
2. Download webpage content.
3. Extract text using BeautifulSoup.
4. Split the text into chunks.
5. Generate embeddings.
6. Store embeddings in FAISS.
7. Receive a user question.
8. Use conversation history to rewrite follow-up questions.
9. Perform Basic, MMR, and Multi-Query retrieval.
10. Compare retrieval results.
11. Build context from retrieved documents.
12. Send the context and conversation history to the Groq language model.
13. Generate the final answer.
14. Save the new question and answer in conversation history.
15. Display the retrieved sources.

## Technologies

- Python
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Groq
- BeautifulSoup
- Requests

## Embedding Model

The application uses:

`sentence-transformers/all-MiniLM-L6-v2`

The model converts document chunks and user queries into vector embeddings that can be searched using FAISS.

## Language Model

The application uses:

`llama-3.3-70b-versatile`

through the Groq API.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
