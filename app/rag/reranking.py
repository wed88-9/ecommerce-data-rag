import os

from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5


# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

def rerank_results(query, hybrid_results, top_k=TOP_K):

    print("\n" + "=" * 60)
    print("CROSS-ENCODER RERANKING")
    print("=" * 60)

    print(f"\nQuery:")
    print(query)

    print(
        f"\nLoading Cross-Encoder model:"
    )
    print(MODEL_NAME)

    model = CrossEncoder(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Create query-document pairs
    # --------------------------------------------------------

    pairs = []

    for result in hybrid_results:

        pairs.append(
            [
                query,
                result["text"],
            ]
        )

    if not pairs:
        print(
            "\nNo hybrid results available."
        )

        return []

    # --------------------------------------------------------
    # Calculate relevance scores
    # --------------------------------------------------------

    print(
        "\nCalculating relevance scores..."
    )

    scores = model.predict(
        pairs
    )

    # --------------------------------------------------------
    # Attach scores
    # --------------------------------------------------------

    reranked_results = []

    for result, score in zip(
        hybrid_results,
        scores,
    ):

        reranked_result = result.copy()

        reranked_result[
            "cross_encoder_score"
        ] = float(score)

        reranked_results.append(
            reranked_result
        )

    # --------------------------------------------------------
    # Sort by Cross-Encoder score
    # --------------------------------------------------------

    reranked_results.sort(
        key=lambda result:
        result["cross_encoder_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Keep top K
    # --------------------------------------------------------

    reranked_results = (
        reranked_results[:top_k]
    )

    # --------------------------------------------------------
    # Update final ranks
    # --------------------------------------------------------

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        result["rank"] = rank

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RERANKED RESULTS")
    print("=" * 60)

    for result in reranked_results:

        print(
            f"\nFinal Rank: "
            f"{result['rank']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Cross-Encoder Score: "
            f"{result['cross_encoder_score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Text:\n"
            f"{result['text'][:500]}"
        )

    return reranked_results


# ============================================================
# DEMO DATA
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("CROSS-ENCODER RERANKING TEST")
    print("=" * 60)

    query = (
        "What is the return policy for phones?"
    )

    # --------------------------------------------------------
    # These represent the results returned by
    # the previous Hybrid Search + RRF stage.
    # --------------------------------------------------------

    hybrid_results = [

        {
            "rank": 1,
            "chunk_id": "chunk_0",
            "source": "ecommerce_knowledge.txt",
            "text": (
                "Phones can be returned within "
                "14 days if unused and in original "
                "condition."
            ),
            "rrf_score": 0.032,
        },

        {
            "rank": 2,
            "chunk_id": "chunk_1",
            "source": "ecommerce_knowledge.txt",
            "text": (
                "All phones include a one year "
                "warranty. Customers should contact "
                "support for warranty claims."
            ),
            "rrf_score": 0.030,
        },

        {
            "rank": 3,
            "chunk_id": "chunk_2",
            "source": "ecommerce_knowledge.txt",
            "text": (
                "Pending orders can be cancelled "
                "before shipment."
            ),
            "rrf_score": 0.028,
        },

        {
            "rank": 4,
            "chunk_id": "chunk_3",
            "source": "ecommerce_knowledge.txt",
            "text": (
                "Customers can contact customer "
                "support for questions about orders."
            ),
            "rrf_score": 0.026,
        },

        {
            "rank": 5,
            "chunk_id": "chunk_4",
            "source": "ecommerce_knowledge.txt",
            "text": (
                "Orders that have already shipped "
                "cannot be cancelled directly."
            ),
            "rrf_score": 0.024,
        },
    ]

    # --------------------------------------------------------
    # Run Cross-Encoder
    # --------------------------------------------------------

    reranked_results = rerank_results(
        query=query,
        hybrid_results=hybrid_results,
        top_k=TOP_K,
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-ENCODER RERANKING COMPLETED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()