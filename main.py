import os
import csv
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

INDEX_PATH = "faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


if os.path.exists(INDEX_PATH):

    print("Loading existing Vector Store...")

    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    print("Building Vector Store...")

    documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    with open("human_rights_links-2.csv", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            try:

                response = requests.get(row["URL"], timeout=10)

                soup = BeautifulSoup(response.text, "html.parser")

                text = soup.get_text(separator=" ", strip=True)

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

            except Exception as e:

                print(f"Skipped: {row['Title']} ({e})")

    print(f"\nTotal Documents: {len(documents)}")

    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(INDEX_PATH)

    print("\nVector Store Saved!")

print("\nSimilarity Search Engine Ready!")

while True:

    query = input("\nEnter your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    results = vector_store.similarity_search(query, k=3)

    print("\nTop Results:\n")

    for i, result in enumerate(results, start=1):

        print("=" * 70)
        print(f"Result {i}")
        print("=" * 70)

        print(result.page_content[:500])
        print("...")

        print("\nTitle:", result.metadata["title"])
        print("URL:", result.metadata["url"])
        # Read all URLs from CSV
# Split page into chunks
# Search for the most relevant chunks