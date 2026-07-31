# Similarity Search Engine with RAG

This project extends a similarity search engine into a complete Retrieval-Augmented Generation (RAG) application.

Instead of only retrieving the most relevant document chunks, the application now sends the retrieved context to a Large Language Model (LLM) through the Groq API to generate a final answer grounded in the retrieved sources.

## Features

- Read source URLs from a CSV file.
- Extract webpage text using BeautifulSoup.
- Split documents into chunks.
- Generate embeddings using HuggingFace.
- Store embeddings in a FAISS vector database.
- Perform semantic similarity search.
- Retrieve the Top-3 most relevant document chunks.
- Send the retrieved context and user question to the Groq API.
- Generate a final answer using Llama 3.3.
- Display the original source titles and URLs.

## Technologies Used

- Python
- FAISS
- LangChain
- HuggingFace Embeddings
- BeautifulSoup
- Requests
- Groq API
- Llama 3.3 70B Versatile

## Project Structure

```
similarity-search-engine/
│
├── main.py
├── human_rights_links-2.csv
├── requirements.txt
├── .gitignore
└── .env (not included)
```

## How It Works

1. Load or build the FAISS vector database.
2. User enters a question.
3. Retrieve the Top-3 most relevant document chunks.
4. Combine the retrieved chunks into a context.
5. Send the context and the user's question to the Groq LLM.
6. Generate the final answer.
7. Display the retrieved sources.

## RAG Pipeline

```
User Question
      ↓
FAISS Similarity Search
      ↓
Top-3 Document Chunks
      ↓
Context Creation
      ↓
Groq API (Llama 3.3)
      ↓
Final Answer
```

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

```
Question:
What is the right to education?

↓

Answer:
The application retrieves the most relevant document chunks,
sends them to the Llama model through Groq,
and generates a grounded response together with the original sources.
```

## Future Improvements

- Support multiple LLM providers.
- Add a web interface.
- Stream responses.
- Support PDF documents.
- Retrieve more configurable Top-K results.

## Author

Ibrahim Blih
