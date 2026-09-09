import json

from confluent_kafka import Producer


KAFKA_CONFIG = {
    "bootstrap.servers": "kafka:9092",
}


producer = Producer(KAFKA_CONFIG)


orders = [
    {
        "order_id": "ORD-001",
        "customer_id": "CUST-001",
        "product": "Laptop",
        "quantity": 2,
        "price": 3500.0,
        "status": "completed",
        "timestamp": "2026-09-08T10:00:00",
    },
    {
        "order_id": "ORD-002",
        "customer_id": "CUST-002",
        "product": "Phone",
        "quantity": 1,
        "price": 2500.0,
        "status": "pending",
        "timestamp": "2026-09-08T10:05:00",
    },
    {
        "order_id": "ORD-003",
        "customer_id": "CUST-003",
        "product": "Tablet",
        "quantity": 0,
        "price": 1800.0,
        "status": "pending",
        "timestamp": "2026-09-08T10:10:00",
    },
]


def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")
    else:
        print(f"Sent order: {msg.key().decode('utf-8')}")


for order in orders:
    producer.produce(
        topic="orders",
        key=order["order_id"].encode("utf-8"),
        value=json.dumps(order).encode("utf-8"),
        callback=delivery_report,
    )

    producer.poll(0)


producer.flush()

print("Producer finished successfully.")