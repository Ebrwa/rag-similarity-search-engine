import os
import csv
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Docum
INDEX_PATH = "faiss_index"
CSV_PATH = "human_rights_links-2.csv"
MODEL_NAME = "llama-3.3-70b-versatile"


# Load variables from the .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Add it to the .env file."
    )

# Create a client for communicating with Groq
client = Groq(api_key=api_key)


# Create the embedding model
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


def generate_answer(question, context):
    """
    Send the user's question and the retrieved context
    to the Groq language model.
    """

    system_message = (
        "You are a helpful question-answering assistant. "
        "Answer the user's question using only the provided context. "
        "Do not use outside knowledge. "
        "If the context does not contain enough information, say that "
        "you could not find enough information in the provided sources. "
        "Give a clear and concise answer."
    )

    user_message = (
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
    Run the complete RAG pipeline.
    """

    vector_store = load_vector_store()

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
            # Retrieval step
            results = vector_store.similarity_search(
                query,
                k=3
            )

            if not results:
                print("No relevant documents were found.")
                continue

            # Augmentation step
            context = create_context(results)

            # Generation step
            answer = generate_answer(
                query,
                context
            )

            print("\nFinal Answer:\n")
            print(answer)

            print_sources(results)

        except Exception as error:
            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()