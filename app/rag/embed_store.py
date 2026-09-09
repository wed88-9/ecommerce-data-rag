import os
import json

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

CHUNKS_FILE = "/app/app/rag/data/chunks.jsonl"

CHROMA_DIR = "/app/app/rag/data/chroma"

COLLECTION_NAME = "ecommerce_knowledge"

# Lightweight and commonly used sentence embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# Load chunks
# ============================================================

def load_chunks(path):

    chunks = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():

                chunks.append(
                    json.loads(line)
                )

    return chunks


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG EMBEDDINGS + CHROMADB")
    print("=" * 60)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    print(
        f"\nLoading chunks from:\n{CHUNKS_FILE}"
    )

    chunks = load_chunks(
        CHUNKS_FILE
    )

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    if not chunks:

        raise RuntimeError(
            "No chunks found."
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        f"\nLoading embedding model:\n"
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded successfully."
    )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"\nCreating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(
        f"Embeddings created."
    )

    print(
        f"Embedding dimension: "
        f"{len(embeddings[0])}"
    )

    # --------------------------------------------------------
    # Create persistent ChromaDB
    # --------------------------------------------------------

    print(
        f"\nCreating ChromaDB at:\n"
        f"{CHROMA_DIR}"
    )

    os.makedirs(
        CHROMA_DIR,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # --------------------------------------------------------
    # Create / reset collection
    # --------------------------------------------------------

    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Existing collection deleted."
        )

    except Exception:

        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
                "E-commerce RAG knowledge base"
        }
    )

    # --------------------------------------------------------
    # Prepare metadata
    # --------------------------------------------------------

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Store embeddings in ChromaDB
    # --------------------------------------------------------

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )

    print(
        "\nEmbeddings stored in ChromaDB."
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Documents stored: "
        f"{collection.count()}"
    )

    # --------------------------------------------------------
    # Test vector search
    # --------------------------------------------------------

    test_query = "How much does the laptop cost?"

    print(
        "\nTesting vector similarity search..."
    )

    query_embedding = model.encode(
        [test_query],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    print(
        f"\nQuery: {test_query}"
    )

    print("-" * 60)

    for i, document in enumerate(
        results["documents"][0]
    ):

        metadata = results["metadatas"][0][i]

        distance = results["distances"][0][i]

        print(
            f"\nResult {i + 1}"
        )

        print(
            f"Chunk: "
            f"{metadata['chunk_id']}"
        )

        print(
            f"Source: "
            f"{metadata['source']}"
        )

        print(
            f"Distance: "
            f"{distance:.4f}"
        )

        print(
            f"Text: "
            f"{document[:300]}"
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EMBEDDINGS + CHROMADB COMPLETED")
    print("=" * 60)