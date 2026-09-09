from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql.functions import col


# ============================================================
# 1. Paths
# ============================================================

bronze_path = "./data/delta/bronze/orders"
silver_path = "./data/delta/silver/orders"


# ============================================================
# 2. Create Spark Session with Delta Lake
# ============================================================

builder = (
    SparkSession.builder
    .appName("EcommerceSilver")
    .config(
        "spark.sql.extensions",
        "io.delta.sql.DeltaSparkSessionExtension"
    )
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

print("Spark + Delta started successfully")


# ============================================================
# 3. Read Bronze Delta Table
# ============================================================

bronze_df = (
    spark.read
    .format("delta")
    .load(bronze_path)
)

print("\nBronze data:")
bronze_df.show(truncate=False)


# ============================================================
# 4. Silver transformations
# ============================================================

silver_df = (
    bronze_df
    .withColumn(
        "total_amount",
        col("quantity") * col("price")
    )
    .dropDuplicates(["order_id"])
)

print("\nTransformed Silver data:")
silver_df.show(truncate=False)


# ============================================================
# 5. Create Silver table if it doesn't exist
# ============================================================

if not DeltaTable.isDeltaTable(spark, silver_path):

    print("\nSilver table does not exist.")
    print("Creating Silver Delta table...")

    (
        silver_df.write
        .format("delta")
        .mode("overwrite")
        .save(silver_path)
    )

    print("Silver Delta table created successfully.")


# ============================================================
# 6. REAL MERGE / UPSERT
# ============================================================

else:

    print("\nSilver table already exists.")
    print("Running REAL Delta MERGE...")

    silver_table = DeltaTable.forPath(
        spark,
        silver_path
    )

    (
        silver_table.alias("target")
        .merge(
            silver_df.alias("source"),
            "target.order_id = source.order_id"
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

    print("MERGE completed successfully.")


# ============================================================
# 7. Read Silver table back
# ============================================================

silver_check = (
    spark.read
    .format("delta")
    .load(silver_path)
)

print("\nSilver Delta Table contents:")
silver_check.show(truncate=False)


# ============================================================
# 8. Test Delta Lake Schema Enforcement
# ============================================================

print("\nTesting Delta Lake schema enforcement...")

bad_data = [
    {
        "customer_id": "CUST-999",
        "order_id": "ORD-BAD",
        "price": "NOT_A_NUMBER",
        "product": "Keyboard",
        "quantity": 1,
        "status": "pending",
        "timestamp": "2026-09-09T12:00:00",
        "total_amount": 100.0
    }
]

bad_df = spark.createDataFrame(bad_data)

try:

    (
        bad_df.write
        .format("delta")
        .mode("append")
        .save(silver_path)
    )

    print("ERROR: Bad data was accepted!")

except Exception:
    print("Schema enforcement worked!")
    print("Rejected bad data.")


# ============================================================
# 9. Stop Spark
# ============================================================

spark.stop()

print("\nSpark stopped.")