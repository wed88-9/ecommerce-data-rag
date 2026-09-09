from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


# ============================================================
# Helper
# ============================================================

def stage_command(command):
    return (
        "docker exec ecommerce-python "
        f"python {command}"
    )


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="ecommerce_data_pipeline",
    start_date=datetime(2026, 9, 9),
    schedule=None,
    catchup=False,
    tags=["ecommerce", "capstone", "rag"],
) as dag:

    # ========================================================
    # 1. Kafka Producer
    # ========================================================

    producer = BashOperator(
        task_id="producer",
        bash_command=(
            "docker exec ecommerce-python "
            "python -m app.ingestion.producer"
        ),
    )

    # ========================================================
    # 2. Kafka Consumer + Pydantic Validation
    # ========================================================

    ingestion = BashOperator(
        task_id="ingestion",
        bash_command=(
            "docker exec ecommerce-python "
            "python -m app.ingestion.consumer"
        ),
    )

    # ========================================================
    # 3. Bronze Layer
    # ========================================================

    bronze = BashOperator(
        task_id="bronze",
        bash_command=stage_command(
            "app/lakehouse/bronze.py"
        ),
    )

    # ========================================================
    # 4. Silver Layer + MERGE + Schema Enforcement
    # ========================================================

    silver = BashOperator(
        task_id="silver",
        bash_command=stage_command(
            "app/lakehouse/silver.py"
        ),
    )

    # ========================================================
    # 5. Great Expectations Quality Gate
    # ========================================================

    quality_gate = BashOperator(
        task_id="quality_gate",
        bash_command=stage_command(
            "app/quality/quality_gate.py"
        ),
    )

    # ========================================================
    # 6. Gold Aggregation
    # ========================================================

    gold = BashOperator(
        task_id="gold",
        bash_command=stage_command(
            "app/lakehouse/gold.py"
        ),
    )

    # ========================================================
    # 7. RAG Chunking
    # ========================================================

    rag_chunking = BashOperator(
        task_id="rag_chunking",
        bash_command=stage_command(
            "app/rag/chunking.py"
        ),
    )

    # ========================================================
    # 8. Embeddings + ChromaDB
    # ========================================================

    rag_embeddings = BashOperator(
        task_id="rag_embeddings",
        bash_command=stage_command(
            "app/rag/embed_store.py"
        ),
    )

    # ========================================================
    # 9. BM25 Keyword Search
    # ========================================================

    rag_bm25 = BashOperator(
        task_id="rag_bm25",
        bash_command=stage_command(
            "app/rag/bm25_search.py"
        ),
    )

    # ========================================================
    # 10. Hybrid Search + RRF
    # ========================================================

    rag_hybrid = BashOperator(
        task_id="rag_hybrid",
        bash_command=stage_command(
            "app/rag/rrf.py"
        ),
    )

    # ========================================================
    # 11. Cross-Encoder Reranking
    # ========================================================

    rag_reranking = BashOperator(
        task_id="rag_reranking",
        bash_command=stage_command(
            "app/rag/reranking.py"
        ),
    )

    # ========================================================
    # 12. Grounded Answer + Citations
    # ========================================================

    rag_answer = BashOperator(
        task_id="rag_answer",
        bash_command=stage_command(
            "app/rag/answer.py"
        ),
    )

    # ========================================================
    # 13. OpenLineage
    # ========================================================

    lineage = BashOperator(
        task_id="lineage",
        bash_command=stage_command(
            "app/lineage/lineage_events.py"
        ),
    )

    # ========================================================
    # PIPELINE DEPENDENCIES
    # ========================================================

    (
        producer
        >> ingestion
        >> bronze
        >> silver
        >> quality_gate
        >> gold
        >> rag_chunking
        >> rag_embeddings
        >> rag_bm25
        >> rag_hybrid
        >> rag_reranking
        >> rag_answer
        >> lineage
    )