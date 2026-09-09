import great_expectations as gx

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


# ============================================================
# 1. Paths
# ============================================================

silver_path = "./data/delta/silver/orders"


# ============================================================
# 2. Start Spark + Delta
# ============================================================

builder = (
    SparkSession.builder
    .appName("EcommerceQualityGate")
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
# 3. Read Silver Delta table
# ============================================================

silver_df = (
    spark.read
    .format("delta")
    .load(silver_path)
)

print("\nSilver data:")
silver_df.show(truncate=False)


# ============================================================
# 4. Start Great Expectations
# ============================================================

print("\nStarting Great Expectations...")

context = gx.get_context()


# ============================================================
# 5. Create Spark Data Source
# ============================================================

data_source_name = "ecommerce_spark_source"

try:
    data_source = context.data_sources.get(data_source_name)
    print("Great Expectations Data Source already exists.")

except Exception:
    data_source = context.data_sources.add_spark(
        name=data_source_name
    )
    print("Great Expectations Spark Data Source created.")


# ============================================================
# 6. Create Data Asset
# ============================================================

data_asset_name = "silver_orders"

try:
    data_asset = data_source.get_asset(data_asset_name)
    print("Great Expectations Data Asset already exists.")

except Exception:
    data_asset = data_source.add_dataframe_asset(
        name=data_asset_name
    )
    print("Great Expectations Data Asset created.")


# ============================================================
# 7. Create Batch Definition
# ============================================================

batch_definition_name = "silver_orders_batch"

try:
    batch_definition = data_asset.get_batch_definition(
        batch_definition_name
    )
    print("Great Expectations Batch Definition already exists.")

except Exception:
    batch_definition = (
        data_asset
        .add_batch_definition_whole_dataframe(
            batch_definition_name
        )
    )

    print("Great Expectations Batch Definition created.")


# ============================================================
# 8. Create Batch from current Silver DataFrame
# ============================================================

batch = batch_definition.get_batch(
    batch_parameters={
        "dataframe": silver_df
    }
)

print("\nGreat Expectations batch created successfully.")


# ============================================================
# 9. Define Expectations
# ============================================================

print("\nRunning Great Expectations checks...")


# Check 1:
# order_id must not be NULL
expect_order_id = gx.expectations.ExpectColumnValuesToNotBeNull(
    column="order_id"
)


# Check 2:
# quantity must be between 1 and 1000
expect_quantity = gx.expectations.ExpectColumnValuesToBeBetween(
    column="quantity",
    min_value=1,
    max_value=1000
)


# Check 3:
# price must be greater than 0
expect_price = gx.expectations.ExpectColumnValuesToBeBetween(
    column="price",
    min_value=0.01,
    max_value=1000000
)


# ============================================================
# 10. Validate expectations
# ============================================================

results = []

results.append(
    batch.validate(expect_order_id)
)

results.append(
    batch.validate(expect_quantity)
)

results.append(
    batch.validate(expect_price)
)


# ============================================================
# 11. Print results
# ============================================================

print("\n============================================================")
print("GREAT EXPECTATIONS RESULTS")
print("============================================================")

all_passed = True

for index, result in enumerate(results, start=1):

    success = result.success

    print(
        f"Expectation {index}: "
        f"{'PASSED' if success else 'FAILED'}"
    )

    if not success:
        all_passed = False

    print(result)


# ============================================================
# 12. Quality Gate
# ============================================================

print("\n============================================================")
print("QUALITY GATE")
print("============================================================")

if all_passed:

    print("QUALITY GATE PASSED")
    print("All Great Expectations checks passed.")

    spark.stop()

    print("Spark stopped.")

else:

    print("QUALITY GATE FAILED")
    print("Great Expectations detected invalid data.")

    spark.stop()

    print("Spark stopped.")

    # IMPORTANT:
    # Non-zero exit code makes Airflow fail this task.
    raise SystemExit(1)