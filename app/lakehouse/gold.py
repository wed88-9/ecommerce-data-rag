from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    countDistinct,
    sum as spark_sum
)
from delta import configure_spark_with_delta_pip


# ============================================================
# 1. Paths
# ============================================================

silver_path = "./data/delta/silver/orders"
gold_path = "./data/delta/gold/product_sales"


# ============================================================
# 2. Create Spark Session with Delta Lake
# ============================================================

builder = (
    SparkSession.builder
    .appName("EcommerceGold")
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
# 3. Read Silver Delta Table
# ============================================================

silver_df = (
    spark.read
    .format("delta")
    .load(silver_path)
)

print("\nSilver data:")
silver_df.show(truncate=False)


# ============================================================
# 4. Create Gold Aggregate
# ============================================================

gold_df = (
    silver_df
    .groupBy("product")
    .agg(
        countDistinct("order_id").alias("total_orders"),
        spark_sum("quantity").alias("total_quantity"),
        spark_sum("total_amount").alias("total_revenue")
    )
)

print("\nGold aggregate:")
gold_df.show(truncate=False)


# ============================================================
# 5. Write Gold as Delta Table
# ============================================================

(
    gold_df.write
    .format("delta")
    .mode("overwrite")
    .save(gold_path)
)

print("\nGold Delta Table written successfully")


# ============================================================
# 6. Read Gold Delta Table Back
# ============================================================

gold_check = (
    spark.read
    .format("delta")
    .load(gold_path)
)

print("\nGold Delta Table contents:")
gold_check.show(truncate=False)


# ============================================================
# 7. Stop Spark
# ============================================================

spark.stop()

print("\nSpark stopped.")