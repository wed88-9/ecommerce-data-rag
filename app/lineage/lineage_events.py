from datetime import datetime, timezone

from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import (
    Dataset,
    Job,
    Run,
    RunEvent,
    RunState,
)
from openlineage.client.uuid import generate_new_uuid
from openlineage.client.transport.file import FileConfig, FileTransport


# --------------------------------------------------
# OpenLineage File Transport
# --------------------------------------------------

file_config = FileConfig(
    log_file_path="/app/data/lineage/events.jsonl",
    append=True,
)

client = OpenLineageClient(
    transport=FileTransport(file_config)
)


# --------------------------------------------------
# Project information
# --------------------------------------------------

PRODUCER = "ecommerce-data-rag"

job = Job(
    namespace="ecommerce-data-rag",
    name="silver-to-gold"
)

run = Run(
    runId=str(generate_new_uuid())
)


# --------------------------------------------------
# Input and Output datasets
# --------------------------------------------------

input_dataset = Dataset(
    namespace="ecommerce-data-rag",
    name="silver_orders"
)

output_dataset = Dataset(
    namespace="ecommerce-data-rag",
    name="gold_product_summary"
)


# --------------------------------------------------
# START event
# --------------------------------------------------

client.emit(
    RunEvent(
        eventType=RunState.START,
        eventTime=datetime.now(timezone.utc).isoformat(),
        run=run,
        job=job,
        producer=PRODUCER,
        inputs=[input_dataset],
    )
)

print("LINEAGE START emitted")


# --------------------------------------------------
# COMPLETE event
# --------------------------------------------------

client.emit(
    RunEvent(
        eventType=RunState.COMPLETE,
        eventTime=datetime.now(timezone.utc).isoformat(),
        run=run,
        job=job,
        producer=PRODUCER,
        inputs=[input_dataset],
        outputs=[output_dataset],
    )
)

print("LINEAGE COMPLETE emitted")