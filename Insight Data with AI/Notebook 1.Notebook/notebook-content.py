# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "acb2cff2-831c-4a84-a2bf-9e49ee48328d",
# META       "default_lakehouse_name": "AzureDatabricks",
# META       "default_lakehouse_workspace_id": "0b00d2c0-b0ef-4be3-93e9-b46cf35f9624",
# META       "known_lakehouses": [
# META         {
# META           "id": "acb2cff2-831c-4a84-a2bf-9e49ee48328d"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Sales order data exploration


# CELL ********************

from pyspark.sql.types import *

# Create the schema for the table
orderSchema = StructType([
    StructField("SalesOrderNumber", StringType()),
    StructField("SalesOrderLineNumber", IntegerType()),
    StructField("OrderDate", DateType()),
    StructField("CustomerName", StringType()),
    StructField("Email", StringType()),
    StructField("Item", StringType()),
    StructField("Quantity", IntegerType()),
    StructField("UnitPrice", FloatType()),
    StructField("Tax", FloatType())
])

# Import all files from bronze folder of lakehouse
df = spark.read.format("csv").option("header", "true").schema(orderSchema).load("Files/bronze/*.csv")

# Display the first 10 rows of the dataframe to preview your data
display(df.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import when, lit, col, current_timestamp, input_file_name

# Add columns IsFlagged, CreatedTS and ModifiedTS
df = df.withColumn("FileName", input_file_name()) \
    .withColumn("IsFlagged", when(col("OrderDate") < '2019-08-01',True).otherwise(False)) \
    .withColumn("CreatedTS", current_timestamp()).withColumn("ModifiedTS", current_timestamp())

# Update CustomerName to "Unknown" if CustomerName null or empty
df = df.withColumn("CustomerName", when((col("CustomerName").isNull() | (col("CustomerName")=="")),lit("Unknown")).otherwise(col("CustomerName")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import *
from delta.tables import *

# Define the schema for the sales_silver table
silver_table_schema = StructType([
    StructField("SalesOrderNumber", StringType(), True),
    StructField("SalesOrderLineNumber", IntegerType(), True),
    StructField("OrderDate", DateType(), True),
    StructField("CustomerName", StringType(), True),
    StructField("Email", StringType(), True),
    StructField("Item", StringType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("UnitPrice", FloatType(), True),
    StructField("Tax", FloatType(), True),
    StructField("FileName", StringType(), True),
    StructField("IsFlagged", BooleanType(), True),
    StructField("CreatedTS", TimestampType(), True),
    StructField("ModifiedTS", TimestampType(), True)
])

# Create or replace the sales_silver table with the defined schema
DeltaTable.createIfNotExists(spark) \
    .tableName("AzureDatabricks.sales_silver") \
    .addColumns(silver_table_schema) \
    .execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import *
from pyspark.sql.functions import when, lit, col, current_timestamp, input_file_name
from delta.tables import *

# Define the schema for the source data
orderSchema = StructType([
    StructField("SalesOrderNumber", StringType(), True),
    StructField("SalesOrderLineNumber", IntegerType(), True),
    StructField("OrderDate", DateType(), True),
    StructField("CustomerName", StringType(), True),
    StructField("Email", StringType(), True),
    StructField("Item", StringType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("UnitPrice", FloatType(), True),
    StructField("Tax", FloatType(), True)
])

# Read data from the bronze folder into a DataFrame
df = spark.read.format("csv").option("header", "true").schema(orderSchema).load("Files/bronze/*.csv")

# Add additional columns
df = df.withColumn("FileName", input_file_name()) \
    .withColumn("IsFlagged", when(col("OrderDate") < '2019-08-01', True).otherwise(False)) \
    .withColumn("CreatedTS", current_timestamp()) \
    .withColumn("ModifiedTS", current_timestamp()) \
    .withColumn("CustomerName", when((col("CustomerName").isNull()) | (col("CustomerName") == ""), lit("Unknown")).otherwise(col("CustomerName")))

# Define the path to the Delta table
deltaTablePath = "Tables/sales_silver"

# Create a DeltaTable object for the existing Delta table
deltaTable = DeltaTable.forPath(spark, deltaTablePath)

# Perform the merge (upsert) operation
deltaTable.alias('silver') \
    .merge(
        df.alias('updates'),
        'silver.SalesOrderNumber = updates.SalesOrderNumber AND \
        silver.OrderDate = updates.OrderDate AND \
        silver.CustomerName = updates.CustomerName AND \
        silver.Item = updates.Item'
    ) \
    .whenMatchedUpdate(set = {
        "SalesOrderLineNumber": "updates.SalesOrderLineNumber",
        "Email": "updates.Email",
        "Quantity": "updates.Quantity",
        "UnitPrice": "updates.UnitPrice",
        "Tax": "updates.Tax",
        "FileName": "updates.FileName",
        "IsFlagged": "updates.IsFlagged",
        "ModifiedTS": "current_timestamp()"
    }) \
    .whenNotMatchedInsert(values = {
        "SalesOrderNumber": "updates.SalesOrderNumber",
        "SalesOrderLineNumber": "updates.SalesOrderLineNumber",
        "OrderDate": "updates.OrderDate",
        "CustomerName": "updates.CustomerName",
        "Email": "updates.Email",
        "Item": "updates.Item",
        "Quantity": "updates.Quantity",
        "UnitPrice": "updates.UnitPrice",
        "Tax": "updates.Tax",
        "FileName": "updates.FileName",
        "IsFlagged": "updates.IsFlagged",
        "CreatedTS": "current_timestamp()",
        "ModifiedTS": "current_timestamp()"
    }) \
    .execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
