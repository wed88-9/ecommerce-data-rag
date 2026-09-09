from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# ---------------------------------------
# 1. Create Spark + Delta Lake session
# ---------------------------------------

builder = (
    SparkSession.builder
    .appName("EcommerceBronze")
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


# ---------------------------------------
# 2. Read raw JSONL data
# ---------------------------------------

input_path = "/app/data/raw/orders.jsonl"

df = spark.read.json(input_path)

print("Raw data loaded:")
df.show()


# ---------------------------------------
# 3. Write Bronze Delta Table
# ---------------------------------------

bronze_path = "/app/data/delta/bronze/orders"

df.write \
    .format("delta") \
    .mode("overwrite") \
    .save(bronze_path)

print("Bronze Delta Table created successfully!")


# ---------------------------------------
# 4. Read Bronze table back
# ---------------------------------------

bronze_df = spark.read.format("delta").load(bronze_path)

print("Bronze Delta Table contents:")
bronze_df.show()


# ---------------------------------------
# 5. Stop Spark
# ---------------------------------------

spark.stop()