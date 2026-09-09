import json
import os

from confluent_kafka import Consumer
from app.ingestion.schema import Order


KAFKA_CONFIG = {
    "bootstrap.servers": "kafka:9092",
"group.id": "orders-consumer-v2",
    "auto.offset.reset": "earliest",
}


QUARANTINE_FILE = "/app/data/quarantine/orders.jsonl"
RAW_FILE = "/app/data/raw/orders.jsonl"


def quarantine(record, reason):
    os.makedirs(
        os.path.dirname(QUARANTINE_FILE),
        exist_ok=True,
    )

    with open(
        QUARANTINE_FILE,
        "a",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "record": record,
                "reason": reason,
            },
            f,
            ensure_ascii=False,
        )

        f.write("\n")


def save_valid_order(order):
    os.makedirs(
        os.path.dirname(RAW_FILE),
        exist_ok=True,
    )

    with open(
        RAW_FILE,
        "a",
        encoding="utf-8",
    ) as f:

        json.dump(
            order.model_dump(),
            f,
            ensure_ascii=False,
        )

        f.write("\n")


consumer = Consumer(KAFKA_CONFIG)

consumer.subscribe(["orders"])

print("Consumer started. Waiting for orders...")


try:

    empty_polls = 0

    while empty_polls < 5:

        msg = consumer.poll(1.0)

        if msg is None:
            empty_polls += 1
            continue

        empty_polls = 0

        if msg.error():

            print(f"Kafka error: {msg.error()}")
            continue

        record = None

        try:

            record = json.loads(
                msg.value().decode("utf-8")
            )

            order = Order.model_validate(record)

            save_valid_order(order)

            print(
                f"VALID ORDER: {order.order_id}"
            )

        except Exception as e:

            reason = str(e)

            print(
                f"REJECTED ORDER: {reason}"
            )

            quarantine(
                record,
                reason,
            )

finally:

    consumer.close()

    print("Consumer finished.")