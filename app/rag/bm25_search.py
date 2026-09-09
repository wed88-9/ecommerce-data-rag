import json

from rank_bm25 import BM25Okapi


# ============================================================
# Configuration
# ============================================================

CHUNKS_FILE = "/app/app/rag/data/chunks.jsonl"


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
# Build BM25 index
# ============================================================

def build_bm25(chunks):

    tokenized_documents = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


# ============================================================
# Search
# ============================================================

def search_bm25(
    query,
    chunks,
    bm25,
    top_k=5
):

    query_tokens = (
        query.lower().split()
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "chunk_id": chunks[index]["chunk_id"],
                "source": chunks[index]["source"],
                "text": chunks[index]["text"],
                "score": float(scores[index]),
            }
        )

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BM25 KEYWORD SEARCH")
    print("=" * 60)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks(
        CHUNKS_FILE
    )

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Build BM25 index
    # --------------------------------------------------------

    bm25 = build_bm25(
        chunks
    )

    print(
        "BM25 index created successfully."
    )

    # --------------------------------------------------------
    # Test search
    # --------------------------------------------------------

    query = "laptop price"

    print(
        f"\nQuery: {query}"
    )

    print("-" * 60)

    results = search_bm25(
        query=query,
        chunks=chunks,
        bm25=bm25,
        top_k=5
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_id']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"BM25 Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Text: "
            f"{result['text'][:300]}"
        )

    print("\n" + "=" * 60)
    print("BM25 SEARCH COMPLETED")
    print("=" * 60)