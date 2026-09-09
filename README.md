# E-Commerce Data Engineering & RAG Pipeline

## Project Overview

This project is an end-to-end modern data engineering pipeline for an e-commerce use case.

The pipeline demonstrates real-time data ingestion, schema validation, Delta Lakehouse processing, data quality validation, orchestration with Apache Airflow, vector search, hybrid RAG retrieval, reranking, grounded answers with citations, and data lineage.

The project was developed as a capstone project for the Modern Data Engineering for AI Systems training program.

---

## Architecture

```text
Kafka Producer
↓
Pydantic Schema Validation
↓
Kafka Consumer
↓
Quarantine Invalid Records
↓
Delta Bronze
↓
Delta Silver + MERGE
↓
Great Expectations Quality Gate
↓
Delta Gold Aggregation
↓
RAG Chunking
↓
Embeddings + ChromaDB
↓
BM25 Search
↓
Hybrid Search + RRF
↓
Cross-Encoder Reranking
↓
Grounded Answer + Citations
↓
OpenLineage
Technologies
Python 3.11
Apache Kafka
Confluent Kafka
Pydantic
Apache Spark
Delta Lake
Great Expectations
Apache Airflow
ChromaDB
Sentence Transformers
BM25
Reciprocal Rank Fusion (RRF)
Cross-Encoder
OpenLineage
Docker
GitHub
Project Structure
ecommerce-data-rag/
│
├── app/
│   ├── ingestion/
│   │   ├── schema.py
│   │   ├── producer.py
│   │   └── consumer.py
│   │
│   ├── lakehouse/
│   │   ├── bronze.py
│   │   ├── silver.py
│   │   └── gold.py
│   │
│   ├── quality/
│   │   └── quality_gate.py
│   │
│   ├── lineage/
│   │   └── lineage_events.py
│   │
│   └── rag/
│       ├── chunking.py
│       ├── embed_store.py
│       ├── bm25_search.py
│       ├── rrf.py
│       ├── reranking.py
│       ├── answer.py
│       └── documents/
│           └── ecommerce_knowledge.txt
│
├── dags/
│   └── ecommerce_pipeline.py
│
├── data/
│
├── docs/
│   └── run_logs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
1. Data Ingestion

The ingestion layer uses Apache Kafka for real-time order streaming.

A Pydantic data contract validates incoming orders before they enter the data pipeline.

Invalid records are rejected and sent to a quarantine zone with the validation error recorded.

Example validation rules include:

order_id must not be empty.
customer_id must not be empty.
product must not be empty.
quantity must be greater than zero.
price must be greater than zero.
status must be a valid order status.

The ingestion stage demonstrates both valid and malformed records.

2. Delta Lakehouse

The project implements a Bronze, Silver, and Gold Lakehouse architecture using Apache Spark and Delta Lake.

Bronze

Bronze stores the raw validated order records.

Silver

Silver contains cleaned and transformed order data.

A real Delta Lake MERGE operation is used with:

order_id

as the business key.

The Silver layer also demonstrates Delta schema enforcement by attempting to write invalid data and confirming that the bad data is rejected.

Gold

Gold contains business-level aggregations.

The Gold layer aggregates order information by product, including:

Total orders
Total quantity
Total revenue

This makes Gold a genuine analytical aggregation rather than a copy of Silver.

3. Data Quality

Great Expectations is used as the data quality gate.

The quality gate validates the Silver data before downstream processing continues.

Examples of checks include:

order_id must not be null.
quantity must be within the expected range.
price must be within the expected range.

If the quality checks fail, the pipeline stops before downstream stages execute.

4. RAG Pipeline

The project implements a Retrieval-Augmented Generation (RAG) pipeline.

Chunking

The e-commerce knowledge document is divided into smaller chunks to support efficient retrieval.

Embeddings

Sentence Transformers are used to generate vector embeddings for the document chunks.

Vector Store

ChromaDB is used as the vector database for storing and retrieving embeddings.

BM25 Search

BM25 provides keyword-based retrieval.

Hybrid Search

Dense vector search and BM25 keyword search are combined.

Reciprocal Rank Fusion (RRF) is used to combine the rankings from the retrieval methods.

Cross-Encoder Reranking

A Cross-Encoder model is used to rerank the retrieved results based on query-document relevance.

Grounded Answers

The final answer is generated using the retrieved context and includes citations to the source document and retrieved chunks.

5. Orchestration

Apache Airflow is used to orchestrate the complete pipeline.

The DAG connects the main processing stages in dependency order:

Producer
↓
Ingestion
↓
Bronze
↓
Silver
↓
Quality Gate
↓
Gold
↓
RAG Chunking
↓
Embeddings
↓
BM25
↓
Hybrid Search
↓
Reranking
↓
Grounded Answer
↓
Lineage

The pipeline was executed through Airflow and completed successfully.

6. Data Lineage

OpenLineage is used to emit lineage events for pipeline execution.

The project records lineage events including:

START
COMPLETE

Lineage output is stored as JSON events.

7. Failure Path Demonstrations

The project includes failure-path demonstrations required for a production-style pipeline.

Malformed Kafka Record

A malformed order with an invalid quantity is rejected by the Pydantic schema validation and written to the quarantine zone.

Delta Schema Enforcement

An invalid data type is intentionally tested against the Delta Silver schema and is rejected.

Quality Gate

Great Expectations validates the Silver data before Gold processing.

A failed quality gate returns a non-zero exit code and prevents downstream processing.

8. How to Run
Prerequisites
Docker Desktop
Git
Python 3.11+
VS Code
Start the Services

From the project root:

docker compose up -d
Run the Airflow Pipeline

Open Airflow:

http://localhost:8080

Then trigger:

ecommerce_data_pipeline
Run Individual Stages

Examples:

docker exec ecommerce-python python -m app.ingestion.producer
docker exec ecommerce-python python -m app.ingestion.consumer
docker exec ecommerce-python python app/lakehouse/bronze.py
docker exec ecommerce-python python app/lakehouse/silver.py
docker exec ecommerce-python python app/quality/quality_gate.py
docker exec ecommerce-python python app/lakehouse/gold.py
9. Expected Results

A successful pipeline execution demonstrates:

Kafka order production and consumption
Pydantic schema validation
Invalid record quarantine
Delta Bronze, Silver, and Gold layers
Delta MERGE upsert
Delta schema enforcement
Great Expectations quality validation
Gold business aggregation
RAG document chunking
Vector embeddings
ChromaDB vector search
BM25 keyword search
Hybrid retrieval using RRF
Cross-Encoder reranking
Grounded answers with citations
Apache Airflow orchestration
OpenLineage events

Execution logs are stored under:

docs/run_logs/
10. Training Program

Training Program: Modern Data Engineering for AI Systems

Organization: SDAIA Academy

Cohort / Session: September 2026

The project integrates concepts covered throughout the training program, including modern data engineering, real-time data pipelines, vector databases, advanced RAG engineering, orchestration, data quality, and data lineage.
SDAIA Academy GitHub:

https://github.com/SDAIAAcademy