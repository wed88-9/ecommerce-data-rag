import os
import re
from pathlib import Path
from typing import List, Dict


# ============================================================
# CONFIGURATION
# ============================================================


DOCUMENT_PATH = "/app/app/rag/documents/ecommerce_knowledge.txt"

TOP_K = 5


# ============================================================
# DOCUMENT LOADING
# ============================================================

def find_document() -> str:
    """
    Find the knowledge document.

    We try the Docker path first, then a few local paths
    so the same code can also be tested outside Docker.
    """

    possible_paths = [
        "/app/app/rag/documents/ecommerce_knowledge.txt",
        "/app/rag/documents/ecommerce_knowledge.txt",
        "app/rag/documents/ecommerce_knowledge.txt",
        "rag/documents/ecommerce_knowledge.txt",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "\nKnowledge document not found.\n"
        "Expected location:\n"
        "  /app/app/rag/documents/ecommerce_knowledge.txt\n"
    )


def load_document(path: str) -> str:
    """
    Load the ecommerce knowledge document.
    """

    print("\n" + "=" * 70)
    print("LOADING KNOWLEDGE DOCUMENT")
    print("=" * 70)

    print(f"Path: {path}")

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    if not text.strip():
        raise ValueError("Knowledge document is empty.")

    print(f"Document characters: {len(text)}")
    print("Document loaded successfully.")

    return text


# ============================================================
# CHUNKING
# ============================================================

def normalize_text(text: str) -> str:
    """
    Clean unnecessary spaces while preserving paragraphs.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[Dict]:

    """
    Split document into overlapping chunks.

    chunk_size = approximate number of characters
    overlap = characters shared between neighboring chunks
    """

    text = normalize_text(text)

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id:04d}",
                    "source": "ecommerce_knowledge.txt",
                    "text": chunk,
                }
            )

            chunk_id += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# KEYWORD SEARCH
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer used by the keyword search.
    """

    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def keyword_search(
    query: str,
    chunks: List[Dict],
    top_k: int = TOP_K,
) -> List[Dict]:

    """
    Simple BM25-style keyword scoring.

    This provides keyword retrieval for the hybrid RAG pipeline.
    """

    query_terms = tokenize(query)

    if not query_terms:
        return []

    scored = []

    for chunk in chunks:

        chunk_terms = tokenize(chunk["text"])

        score = 0

        for term in query_terms:
            score += chunk_terms.count(term)

        if score > 0:

            result = chunk.copy()
            result["keyword_score"] = float(score)

            scored.append(result)

    scored.sort(
        key=lambda x: x["keyword_score"],
        reverse=True,
    )

    return scored[:top_k]


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(
    query: str,
    chunks: List[Dict],
    top_k: int = TOP_K,
) -> List[Dict]:

    """
    Dense semantic retrieval using Sentence Transformers.

    If embeddings cannot be loaded, the function returns an empty
    result rather than crashing the complete demo.
    """

    try:

        from sentence_transformers import SentenceTransformer
        import numpy as np

    except Exception as exc:

        print(f"WARNING: Dense search unavailable: {exc}")

        return []

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
    )

    scores = np.dot(
        document_embeddings,
        query_embedding,
    )

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:

        result = chunks[index].copy()

        result["dense_score"] = float(
            scores[index]
        )

        results.append(result)

    return results


# ============================================================
# RECIPROCAL RANK FUSION - RRF
# ============================================================

def reciprocal_rank_fusion(
    dense_results: List[Dict],
    keyword_results: List[Dict],
    top_k: int = TOP_K,
) -> List[Dict]:

    """
    Combine dense and keyword rankings using RRF.

    RRF formula:

        score = 1 / (k + rank)

    where k is usually 60.
    """

    RRF_K = 60

    fused = {}

    # --------------------------------------------------------
    # Dense ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        chunk_id = result["chunk_id"]

        if chunk_id not in fused:

            fused[chunk_id] = result.copy()

            fused[chunk_id]["rrf_score"] = 0.0

        fused[chunk_id]["rrf_score"] += (
            1.0 / (RRF_K + rank)
        )

    # --------------------------------------------------------
    # Keyword ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):

        chunk_id = result["chunk_id"]

        if chunk_id not in fused:

            fused[chunk_id] = result.copy()

            fused[chunk_id]["rrf_score"] = 0.0

        fused[chunk_id]["rrf_score"] += (
            1.0 / (RRF_K + rank)
        )

    results = list(fused.values())

    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(
    query: str,
    results: List[Dict],
    top_k: int = TOP_K,
) -> List[Dict]:

    """
    Rerank hybrid search results using a Cross-Encoder.
    """

    if not results:
        return []

    try:

        from sentence_transformers import CrossEncoder

    except Exception as exc:

        print(
            f"WARNING: Cross-Encoder unavailable: {exc}"
        )

        return results[:top_k]

    print("\nLoading Cross-Encoder model...")

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = [
        (
            query,
            result["text"],
        )
        for result in results
    ]

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):

        item = result.copy()

        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    results: List[Dict],
) -> str:

    context_parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        context_parts.append(
            f"""
[Context {index}]
Source: {result['source']}
Chunk ID: {result['chunk_id']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GROUNDED ANSWER
# ============================================================

def generate_grounded_answer(
    query: str,
    results: List[Dict],
) -> str:

    """
    Generate an extractive grounded answer from retrieved context.

    The answer only uses sentences that appear in the retrieved
    context, which keeps the demo grounded.
    """

    if not results:

        return (
            "I could not find relevant information "
            "in the knowledge base."
        )

    query_terms = set(
        tokenize(query)
    )

    candidate_sentences = []

    for result in results:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            result["text"],
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_terms = set(
                tokenize(sentence)
            )

            overlap = len(
                query_terms.intersection(
                    sentence_terms
                )
            )

            if overlap > 0:

                candidate_sentences.append(
                    (
                        overlap,
                        result,
                        sentence,
                    )
                )

    # Highest keyword overlap first
    candidate_sentences.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    if not candidate_sentences:

        # If there is no exact keyword overlap,
        # return the highest ranked retrieved context.
        return results[0]["text"]

    selected = candidate_sentences[:3]

    answer_parts = []

    used = set()

    for _, result, sentence in selected:

        if sentence in used:
            continue

        used.add(sentence)

        answer_parts.append(sentence)

    return " ".join(answer_parts)


# ============================================================
# CITATIONS
# ============================================================

def build_citations(
    results: List[Dict],
) -> List[str]:

    citations = []

    for result in results:

        citation = (
            f"{result['source']} "
            f"({result['chunk_id']})"
        )

        if citation not in citations:

            citations.append(citation)

    return citations


# ============================================================
# RAG PIPELINE
# ============================================================

def run_rag(query: str):

    print("\n" + "=" * 70)
    print("RAG PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load document
    # --------------------------------------------------------

    document_path = find_document()

    document = load_document(
        document_path
    )

    # --------------------------------------------------------
    # 2. Chunking
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHUNKING")
    print("=" * 70)

    chunks = chunk_text(
        document,
        chunk_size=500,
        overlap=80,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # 3. Dense retrieval
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DENSE VECTOR SEARCH")
    print("=" * 70)

    dense_results = dense_search(
        query=query,
        chunks=chunks,
        top_k=TOP_K,
    )

    print(
        f"Dense results: {len(dense_results)}"
    )

    # --------------------------------------------------------
    # 4. Keyword retrieval
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("KEYWORD SEARCH")
    print("=" * 70)

    keyword_results = keyword_search(
        query=query,
        chunks=chunks,
        top_k=TOP_K,
    )

    print(
        f"Keyword results: {len(keyword_results)}"
    )

    # --------------------------------------------------------
    # 5. Hybrid search + RRF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID SEARCH + RRF")
    print("=" * 70)

    hybrid_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        keyword_results=keyword_results,
        top_k=TOP_K,
    )

    print(
        f"Hybrid results: {len(hybrid_results)}"
    )

    for result in hybrid_results:

        print(
            f"\n{result['chunk_id']} "
            f"| RRF score: "
            f"{result['rrf_score']:.6f}"
        )

    # --------------------------------------------------------
    # 6. Cross-Encoder reranking
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-ENCODER RERANKING")
    print("=" * 70)

    reranked_results = rerank_results(
        query=query,
        results=hybrid_results,
        top_k=TOP_K,
    )

    for result in reranked_results:

        print(
            f"\n{result['chunk_id']} "
            f"| Rerank score: "
            f"{result.get('rerank_score', 0):.4f}"
        )

    # --------------------------------------------------------
    # 7. Retrieved context
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    context = build_context(
        reranked_results
    )

    print(context[:5000])

    # --------------------------------------------------------
    # 8. Grounded answer
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GROUNDED ANSWER")
    print("=" * 70)

    answer = generate_grounded_answer(
        query=query,
        results=reranked_results,
    )

    print(answer)

    # --------------------------------------------------------
    # 9. Citations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CITATIONS")
    print("=" * 70)

    citations = build_citations(
        reranked_results
    )

    for citation in citations:

        print(f"- {citation}")

    # --------------------------------------------------------
    # 10. Final evidence
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GROUNDED ANSWER + CITATIONS COMPLETED")
    print("=" * 70)


# ============================================================
# DEMO
# ============================================================

def run_demo():

    print("\n" + "=" * 70)
    print("E-COMMERCE RAG DEMO")
    print("=" * 70)

    query = (
        "What is the return policy for phones?"
    )

    print(
        f"\nQuestion:\n{query}"
    )

    run_rag(query)


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    run_demo()


if __name__ == "__main__":
    main()