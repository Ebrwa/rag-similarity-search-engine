import os
import csv
import time
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


INDEX_PATH = "faiss_index"
CSV_PATH = "human_rights_links-2.csv"
MODEL_NAME = "llama-3.3-70b-versatile"


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Add it to the .env file."
    )


client = Groq(api_key=api_key)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vector_store():
    """
    Read webpages from the CSV file, split their text into chunks,
    convert the chunks into embeddings, and save them in FAISS.
    """

    print("Building Vector Store...")

    documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    with open(CSV_PATH, encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                response = requests.get(
                    row["URL"],
                    timeout=10,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

                response.raise_for_status()

                soup = BeautifulSoup(
                    response.text,
                    "html.parser"
                )

                text = soup.get_text(
                    separator=" ",
                    strip=True
                )

                chunks = text_splitter.split_text(text)

                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "title": row["Title"],
                                "url": row["URL"]
                            }
                        )
                    )

                print(f"Finished: {row['Title']}")

            except Exception as error:
                print(
                    f"Skipped: {row['Title']} ({error})"
                )

    if not documents:
        raise ValueError(
            "No documents were created from the URLs."
        )

    print(f"\nTotal document chunks: {len(documents)}")

    new_vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    new_vector_store.save_local(INDEX_PATH)

    print("Vector Store Saved!")

    return new_vector_store


def load_vector_store():
    """
    Load the saved FAISS vector store if it exists.
    Otherwise, build a new one.
    """

    if os.path.exists(INDEX_PATH):
        print("Loading existing Vector Store...")

        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return build_vector_store()


def rewrite_question(question, conversation_history):
    """
    Rewrite a follow-up question using the previous conversation.
    """

    if not conversation_history:
        return question

    history_text = ""

    for message in conversation_history[-4:]:
        history_text += (
            f"User: {message['user']}\n"
            f"Assistant: {message['assistant']}\n"
        )

    system_message = (
        "Rewrite the latest user question as a standalone question "
        "using the previous conversation. "
        "Do not answer the question. "
        "Return only the rewritten question."
    )

    user_message = (
        f"Conversation:\n{history_text}\n"
        f"Latest question: {question}"
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0,
        max_completion_tokens=150
    )

    return completion.choices[0].message.content.strip()


def basic_retrieval(vector_store, query):
    return vector_store.similarity_search(
        query,
        k=3
    )


def mmr_retrieval(vector_store, query):
    return vector_store.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=10,
        lambda_mult=0.7
    )


def generate_search_queries(question):
    system_message = (
        "Generate three different search queries for this question. "
        "Keep the same meaning but use different wording. "
        "Return one query per line without numbering."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.3,
        max_completion_tokens=150
    )

    text = completion.choices[0].message.content

    queries = []

    for line in text.splitlines():
        query = line.strip().lstrip("-").strip()

        if query:
            queries.append(query)

    return queries[:3]


def multi_query_retrieval(vector_store, question):
    queries = generate_search_queries(question)

    all_results = []

    for query in queries:
        results = vector_store.similarity_search(
            query,
            k=3
        )

        all_results.extend(results)

    unique_results = []
    seen = set()

    for result in all_results:
        if result.page_content not in seen:
            unique_results.append(result)
            seen.add(result.page_content)

    return unique_results[:5]


def print_retrieval_results(name, results, elapsed_time):
    print(f"\n{name}")
    print(f"Time: {elapsed_time:.3f} seconds")
    print(f"Results: {len(results)}")

    unique_sources = set()

    for index, result in enumerate(results, start=1):
        title = result.metadata.get(
            "title",
            "Unknown title"
        )

        url = result.metadata.get(
            "url",
            "Unknown URL"
        )

        unique_sources.add(url)

        print(f"{index}. {title}")

    print(f"Unique sources: {len(unique_sources)}")


def compare_retrieval_methods(vector_store, query):
    print("\nRetrieval comparison:")

    start = time.perf_counter()

    basic_results = basic_retrieval(
        vector_store,
        query
    )

    basic_time = time.perf_counter() - start

    start = time.perf_counter()

    mmr_results = mmr_retrieval(
        vector_store,
        query
    )

    mmr_time = time.perf_counter() - start

    start = time.perf_counter()

    multi_query_results = multi_query_retrieval(
        vector_store,
        query
    )

    multi_query_time = time.perf_counter() - start

    print_retrieval_results(
        "Basic Similarity Search",
        basic_results,
        basic_time
    )

    print_retrieval_results(
        "MMR",
        mmr_results,
        mmr_time
    )

    print_retrieval_results(
        "Multi-Query",
        multi_query_results,
        multi_query_time
    )

    return multi_query_results


def create_context(results):
    """
    Combine the retrieved document chunks into one context
    that can be sent to the LLM.
    """

    context_parts = []

    for index, result in enumerate(results, start=1):
        title = result.metadata.get(
            "title",
            "Unknown title"
        )

        url = result.metadata.get(
            "url",
            "Unknown URL"
        )

        context_part = (
            f"Source {index}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {result.page_content}"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)


def generate_answer(question, context, conversation_history):
    """
    Generate an answer using retrieved documents
    and previous conversation messages.
    """

    history_text = ""

    for message in conversation_history[-4:]:
        history_text += (
            f"User: {message['user']}\n"
            f"Assistant: {message['assistant']}\n"
        )

    system_message = (
        "You are a helpful question-answering assistant. "
        "Answer using only the provided context. "
        "Use the conversation history to understand follow-up questions. "
        "Do not use outside knowledge. "
        "If there is not enough information in the context, say that "
        "you could not find enough information in the provided sources. "
        "Give a clear and concise answer."
    )

    user_message = (
        f"Conversation history:\n{history_text}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.2,
        max_completion_tokens=500
    )

    return completion.choices[0].message.content


def print_sources(results):
    """
    Print the sources retrieved from the vector database.
    """

    print("\nSources:")

    for index, result in enumerate(results, start=1):
        title = result.metadata.get(
            "title",
            "Unknown title"
        )

        url = result.metadata.get(
            "url",
            "Unknown URL"
        )

        print(f"\n{index}. {title}")
        print(url)


def main():
    """
    Run the conversational RAG application.
    """

    vector_store = load_vector_store()

    conversation_history = []

    print("\nRAG Application Ready!")

    while True:
        query = input(
            "\nEnter your question "
            "(type 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            search_query = rewrite_question(
                query,
                conversation_history
            )

            if search_query != query:
                print(
                    f"\nRewritten question: {search_query}"
                )

            results = compare_retrieval_methods(
                vector_store,
                search_query
            )

            if not results:
                print("No relevant documents were found.")
                continue

            context = create_context(results)

            answer = generate_answer(
                query,
                context,
                conversation_history
            )

            print("\nFinal Answer:\n")
            print(answer)

            conversation_history.append(
                {
                    "user": query,
                    "assistant": answer
                }
            )

            print_sources(results)

        except Exception as error:
            print(
                f"\nAn error occurred: {error}"
            )


if __name__ == "__main__":
    main()