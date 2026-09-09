import os
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_PATH = "/app/app/rag/documents/ecommerce_knowledge.txt"

CHROMA_PATH = "/app/data/chroma"

COLLECTION_NAME = "ecommerce_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K_DENSE = 5
TOP_K_BM25 = 5
TOP_K_HYBRID = 5

RRF_K = 60


# ============================================================
# LOAD DOCUMENT
# ============================================================

def load_document(path):
    print("=" * 60)
    print("LOADING DOCUMENT")
    print("=" * 60)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"Document loaded successfully.")
    print(f"Characters: {len(text)}")

    return text


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split document into overlapping text chunks.
    """

    print("\n" + "=" * 60)
    print("CHUNKING DOCUMENT")
    print("=" * 60)

    text = re.sub(r"\s+", " ", text).strip()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": chunk,
                    "source": "ecommerce_knowledge.txt",
                }
            )

            chunk_id += 1

        if end >= len(text):
            break

        start = end - overlap

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ============================================================
# TOKENIZATION FOR BM25
# ============================================================

def tokenize(text):
    """
    Simple tokenizer for BM25 keyword search.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE,
    )


# ============================================================
# CREATE EMBEDDINGS + CHROMADB
# ============================================================

def create_vector_store(chunks):
    print("\n" + "=" * 60)
    print("EMBEDDINGS + CHROMADB")
    print("=" * 60)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print(f"Generated {len(embeddings)} embeddings.")

    os.makedirs(CHROMA_PATH, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Delete old collection if it exists.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Old ChromaDB collection removed.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "E-commerce knowledge vector store"
        },
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print("Documents added to ChromaDB.")
    print("EMBEDDINGS + CHROMADB COMPLETED")

    return model, client, collection


# ============================================================
# DENSE VECTOR SEARCH
# ============================================================

def dense_search(
    query,
    model,
    collection,
    top_k=TOP_K_DENSE,
):
    print("\n" + "=" * 60)
    print("DENSE VECTOR SEARCH")
    print("=" * 60)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    dense_results = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):

        dense_results.append(
            {
                "rank": rank,
                "chunk_id": metadata["chunk_id"],
                "source": metadata["source"],
                "text": document,
                "distance": float(distance),
            }
        )

    print(
        f"Returned {len(dense_results)} dense results."
    )

    for result in dense_results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print(
            f"Text: {result['text'][:300]}"
        )

    return dense_results


# ============================================================
# BM25 KEYWORD SEARCH
# ============================================================

def bm25_search(
    query,
    chunks,
    top_k=TOP_K_BM25,
):
    print("\n" + "=" * 60)
    print("BM25 KEYWORD SEARCH")
    print("=" * 60)

    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    bm25_results = []

    for rank, index in enumerate(
        ranked_indexes[:top_k],
        start=1,
    ):

        chunk = chunks[index]

        bm25_results.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(scores[index]),
            }
        )

    print(
        f"Returned {len(bm25_results)} BM25 results."
    )

    for result in bm25_results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Text: {result['text'][:300]}"
        )

    print("\n" + "=" * 60)
    print("BM25 SEARCH COMPLETED")
    print("=" * 60)

    return bm25_results


# ============================================================
# RECIPROCAL RANK FUSION - RRF
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=TOP_K_HYBRID,
    k=RRF_K,
):
    print("\n" + "=" * 60)
    print("RECIPROCAL RANK FUSION (RRF)")
    print("=" * 60)

    scores = {}

    data = {}

    # --------------------------------------------------------
    # Dense ranking
    # --------------------------------------------------------

    for result in dense_results:

        chunk_id = result["chunk_id"]

        rrf_score = 1.0 / (
            k + result["rank"]
        )

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + rrf_score
        )

        data[chunk_id] = result

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for result in bm25_results:

        chunk_id = result["chunk_id"]

        rrf_score = 1.0 / (
            k + result["rank"]
        )

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + rrf_score
        )

        if chunk_id not in data:
            data[chunk_id] = result

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    ranked_chunks = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    hybrid_results = []

    for final_rank, (
        chunk_id,
        rrf_score,
    ) in enumerate(
        ranked_chunks[:top_k],
        start=1,
    ):

        result = data[chunk_id]

        hybrid_results.append(
            {
                "rank": final_rank,
                "chunk_id": chunk_id,
                "source": result["source"],
                "text": result["text"],
                "rrf_score": rrf_score,
            }
        )

    print(
        f"Returned {len(hybrid_results)} hybrid results."
    )

    for result in hybrid_results:

        print(
            f"\nFinal Rank: {result['rank']}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"RRF Score: {result['rrf_score']:.6f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Text: {result['text'][:300]}"
        )

    print("\n" + "=" * 60)
    print("RRF FUSION COMPLETED")
    print("=" * 60)

    return hybrid_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("HYBRID SEARCH PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load document
    # --------------------------------------------------------

    document = load_document(
        DOCUMENT_PATH
    )

    # --------------------------------------------------------
    # 2. Chunk document
    # --------------------------------------------------------

    chunks = chunk_text(
        document,
        chunk_size=500,
        overlap=100,
    )

    # --------------------------------------------------------
    # Display sample chunks
    # --------------------------------------------------------

    print("\nSample chunks:")
    print("-" * 60)

    for chunk in chunks[:5]:

        print(
            f"\n[{chunk['chunk_id']}]"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            f"Text: {chunk['text'][:250]}"
        )

    # --------------------------------------------------------
    # 3. Create embeddings + ChromaDB
    # --------------------------------------------------------

    model, client, collection = (
        create_vector_store(chunks)
    )

    # --------------------------------------------------------
    # 4. Test query
    # --------------------------------------------------------

    query = (
        "What is the return policy for phones?"
    )

    print("\n" + "=" * 60)
    print("QUERY")
    print("=" * 60)

    print(f"Question: {query}")

    # --------------------------------------------------------
    # 5. Dense search
    # --------------------------------------------------------

    dense_results = dense_search(
        query=query,
        model=model,
        collection=collection,
        top_k=TOP_K_DENSE,
    )

    # --------------------------------------------------------
    # 6. BM25 search
    # --------------------------------------------------------

    bm25_results = bm25_search(
        query=query,
        chunks=chunks,
        top_k=TOP_K_BM25,
    )

    # --------------------------------------------------------
    # 7. Hybrid search using RRF
    # --------------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=TOP_K_HYBRID,
        k=RRF_K,
    )

    # --------------------------------------------------------
    # 8. Final output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL HYBRID RESULTS")
    print("=" * 60)

    for result in hybrid_results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"RRF Score: {result['rrf_score']:.6f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Text:\n{result['text'][:500]}"
        )

    print("\n" + "=" * 60)
    print("HYBRID SEARCH COMPLETED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()