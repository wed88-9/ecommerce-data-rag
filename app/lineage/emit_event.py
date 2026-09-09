import sys
import os
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


# ============================================================
# Configuration
# ============================================================

LINEAGE_DIR = "/app/data/lineage"
LINEAGE_FILE = os.path.join(LINEAGE_DIR, "events.jsonl")

os.makedirs(LINEAGE_DIR, exist_ok=True)

file_config = FileConfig(
    log_file_path=LINEAGE_FILE,
    append=True,
)

client = OpenLineageClient(
    transport=FileTransport(file_config)
)

PRODUCER = "ecommerce-data-rag"


# ============================================================
# Stage configuration
# ============================================================

STAGES = {
    "producer": {
        "inputs": [],
        "outputs": ["kafka_orders"],
    },
    "ingestion": {
        "inputs": ["kafka_orders"],
        "outputs": ["raw_orders"],
    },
    "bronze": {
        "inputs": ["raw_orders"],
        "outputs": ["bronze_orders"],
    },
    "silver": {
        "inputs": ["bronze_orders"],
        "outputs": ["silver_orders"],
    },
    "quality_gate": {
        "inputs": ["silver_orders"],
        "outputs": ["quality_checked_silver"],
    },
    "gold": {
        "inputs": ["silver_orders"],
        "outputs": ["gold_product_summary"],
    },
    "lineage": {
        "inputs": ["silver_orders"],
        "outputs": ["gold_product_summary"],
    },
}


# ============================================================
# Emit Lineage Event
# ============================================================

def emit_event(stage, state):

    if stage not in STAGES:
        raise ValueError(
            f"Unknown stage: {stage}. "
            f"Available stages: {list(STAGES.keys())}"
        )

    if state not in ["START", "COMPLETE", "FAIL"]:
        raise ValueError(
            "State must be START, COMPLETE, or FAIL."
        )

    config = STAGES[stage]

    job = Job(
        namespace="ecommerce-data-rag",
        name=stage,
    )

    run = Run(
        runId=str(generate_new_uuid())
    )

    inputs = [
        Dataset(
            namespace="ecommerce-data-rag",
            name=name,
        )
        for name in config["inputs"]
    ]

    outputs = [
        Dataset(
            namespace="ecommerce-data-rag",
            name=name,
        )
        for name in config["outputs"]
    ]

    event = RunEvent(
        eventType=RunState[state],
        eventTime=datetime.now(timezone.utc).isoformat(),
        run=run,
        job=job,
        producer=PRODUCER,
        inputs=inputs,
        outputs=outputs,
    )

    client.emit(event)

    print(
        f"LINEAGE {state} emitted | "
        f"stage={stage}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python emit_event.py "
            "<stage> <START|COMPLETE|FAIL>"
        )
        sys.exit(1)

    stage = sys.argv[1]
    state = sys.argv[2].upper()

    emit_event(stage, state)