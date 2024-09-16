# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "127c1edf-84ba-4a4a-b94b-434af30716fa",
# META       "default_lakehouse_name": "wwilakehouse",
# META       "default_lakehouse_workspace_id": "35de35eb-5166-4692-ad1e-249ff5e0626c",
# META       "known_lakehouses": [
# META         {
# META           "id": "127c1edf-84ba-4a4a-b94b-434af30716fa"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Load data to the dataframe as a starting point to create the gold layer
df = spark.read.table("wwilakehouse.sales_silver")

# Display the first few rows of the dataframe to verify the data
df.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import *
from delta.tables import*

# Define the schema for the dimdate_gold table
DeltaTable.createIfNotExists(spark) \
    .tableName("wwilakehouse.dimdate_gold") \
    .addColumn("OrderDate", DateType()) \
    .addColumn("Day", IntegerType()) \
    .addColumn("Month", IntegerType()) \
    .addColumn("Year", IntegerType()) \
    .addColumn("mmmyyyy", StringType()) \
    .addColumn("yyyymm", StringType()) \
    .execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, dayofmonth, month, year, date_format

# Create dataframe for dimDate_gold

dfdimDate_gold =df.dropDuplicates(["OrderDate"]).select(col("OrderDate"), \
        dayofmonth("OrderDate").alias("Day"), \
        month("OrderDate").alias("Month"), \
        year("OrderDate").alias("Year"), \
        date_format(col("OrderDate"), "MMM-yyyy").alias("mmmyyyy"), \
        date_format(col("OrderDate"), "yyyyMM").alias("yyyymm"), \
    ).orderBy("OrderDate")

# Display the first 10 rows of the dataframe to preview your data

display(dfdimDate_gold.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, dayofmonth, month, year, date_format

# Create dataframe for dimDate_gold

dfdimDate_gold =df.dropDuplicates(["OrderDate"]).select(col("OrderDate"), \
        dayofmonth("OrderDate").alias("Day"), \
        month("OrderDate").alias("Month"), \
        year("OrderDate").alias("Year"), \
        date_format(col("OrderDate"), "MMM-yyyy").alias("mmmyyyy"), \
        date_format(col("OrderDate"), "yyyyMM").alias("yyyymm"), \
    ).orderBy("OrderDate")

# Display the first 10 rows of the dataframe to preview your data

display(dfdimDate_gold.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, 'Tables/dimdate_gold')

dfUpdates = dfdimDate_gold

deltaTable.alias('silver') \
.merge(
    dfUpdates.alias('updates'),
    'silver.OrderDate = updates.OrderDate'
) \
    .whenMatchedUpdate(set =
    {
}
) \
.whenNotMatchedInsert(values =
    {
    "OrderDate": "updates.OrderDate",
    "Day": "updates.Day",
    "Month": "updates.Month",
    "Year": "updates.Year",
    "mmmyyyy": "updates.mmmyyyy",
    "yyyymm": "yyyymm"
    }
) \
.execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import *
from delta.tables import *

# Create customer_gold dimension delta table
DeltaTable.createIfNotExists(spark) \
    .tableName("wwilakehouse.dimcustomer_gold") \
    .addColumn("CustomerName", StringType()) \
    .addColumn("Email",  StringType()) \
    .addColumn("First", StringType()) \
    .addColumn("Last", StringType()) \
    .addColumn("CustomerID", LongType()) \
    .execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, split

# Create customer_silver dataframe

dfdimCustomer_silver = df.dropDuplicates(["CustomerName","Email"]).select(col("CustomerName"),col("Email")) \
    .withColumn("First",split(col("CustomerName"), " ").getItem(0)) \
    .withColumn("Last",split(col("CustomerName"), " ").getItem(1)) 

# Display the first 10 rows of the dataframe to preview your data

display(dfdimCustomer_silver.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import monotonically_increasing_id, col, when, coalesce, max, lit

dfdimCustomer_temp = spark.read.table("wwilakehouse.dimCustomer_gold")

MAXCustomerID = dfdimCustomer_temp.select(coalesce(max(col("CustomerID")),lit(0)).alias("MAXCustomerID")).first()[0]

dfdimCustomer_gold = dfdimCustomer_silver.join(dfdimCustomer_temp,(dfdimCustomer_silver.CustomerName == dfdimCustomer_temp.CustomerName) & (dfdimCustomer_silver.Email == dfdimCustomer_temp.Email), "left_anti")

dfdimCustomer_gold = dfdimCustomer_gold.withColumn("CustomerID",monotonically_increasing_id() + MAXCustomerID + 1)

# Display the first 10 rows of the dataframe to preview your data

display(dfdimCustomer_gold.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, 'Tables/dimcustomer_gold')

dfUpdates = dfdimCustomer_gold

deltaTable.alias('silver') \
.merge(
    dfUpdates.alias('updates'),
    'silver.CustomerName = updates.CustomerName AND silver.Email = updates.Email'
) \
.whenMatchedUpdate(set =
    {
}
) \
.whenNotMatchedInsert(values =
    {
    "CustomerName": "updates.CustomerName",
    "Email": "updates.Email",
    "First": "updates.First",
    "Last": "updates.Last",
    "CustomerID": "updates.CustomerID"
    }
) \
.execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import *
from delta.tables import *

DeltaTable.createIfNotExists(spark) \
    .tableName("wwilakehouse.dimproduct_gold") \
    .addColumn("ItemName", StringType()) \
    .addColumn("ItemID", LongType()) \
    .addColumn("ItemInfo", StringType()) \
    .execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, split, lit

# Create product_silver dataframe

dfdimProduct_silver = df.dropDuplicates(["Item"]).select(col("Item")) \
    .withColumn("ItemName",split(col("Item"), ", ").getItem(0)) \
    .withColumn("ItemInfo",when((split(col("Item"), ", ").getItem(1).isNull() | (split(col("Item"), ", ").getItem(1)=="")),lit("")).otherwise(split(col("Item"), ", ").getItem(1))) 

# Display the first 10 rows of the dataframe to preview your data

display(dfdimProduct_silver.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import monotonically_increasing_id, col, lit, max, coalesce

#dfdimProduct_temp = dfdimProduct_silver
dfdimProduct_temp = spark.read.table("wwilakehouse.dimProduct_gold")

MAXProductID = dfdimProduct_temp.select(coalesce(max(col("ItemID")),lit(0)).alias("MAXItemID")).first()[0]

dfdimProduct_gold = dfdimProduct_silver.join(dfdimProduct_temp,(dfdimProduct_silver.ItemName == dfdimProduct_temp.ItemName) & (dfdimProduct_silver.ItemInfo == dfdimProduct_temp.ItemInfo), "left_anti")

dfdimProduct_gold = dfdimProduct_gold.withColumn("ItemID",monotonically_increasing_id() + MAXProductID + 1)

# Display the first 10 rows of the dataframe to preview your data

display(dfdimProduct_gold.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, 'Tables/dimproduct_gold')

dfUpdates = dfdimProduct_gold

deltaTable.alias('silver') \
.merge(
        dfUpdates.alias('updates'),
        'silver.ItemName = updates.ItemName AND silver.ItemInfo = updates.ItemInfo'
        ) \
        .whenMatchedUpdate(set =
        {
    }
    ) \
    .whenNotMatchedInsert(values =
    {
    "ItemName": "updates.ItemName",
    "ItemInfo": "updates.ItemInfo",
    "ItemID": "updates.ItemID"
    }
    ) \
    .execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import *
from delta.tables import *

DeltaTable.createIfNotExists(spark) \
    .tableName("wwilakehouse.factsales_gold") \
    .addColumn("CustomerID", LongType()) \
    .addColumn("ItemID", LongType()) \
    .addColumn("OrderDate", DateType()) \
    .addColumn("Quantity", IntegerType()) \
    .addColumn("UnitPrice", FloatType()) \
    .addColumn("Tax", FloatType()) \
    .execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType, BooleanType, TimestampType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DeltaTableUpsert") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

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

# Define the path to the Delta table (ensure this path is correct)
delta_table_path = "abfss://<container>@<storage-account>.dfs.core.windows.net/path/to/wwilakehouse/sales_silver"

# Create a DataFrame with the defined schema
empty_df = spark.createDataFrame([], silver_table_schema)

# Register the Delta table in the Metastore
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS wwilakehouse.sales_silver
    USING DELTA
    LOCATION '{delta_table_path}'
""")

# Load data into DataFrame
df = spark.read.table("wwilakehouse.sales_silver")

# Perform transformations on df
df = df.withColumn("ItemName", split(col("Item"), ", ").getItem(0)) \
    .withColumn("ItemInfo", when(
        (split(col("Item"), ", ").getItem(1).isNull()) | (split(col("Item"), ", ").getItem(1) == ""),
        lit("")
    ).otherwise(split(col("Item"), ", ").getItem(1)))

# Load additional DataFrames for joins
dfdimCustomer_temp = spark.read.table("wwilakehouse.dimCustomer_gold")
dfdimProduct_temp = spark.read.table("wwilakehouse.dimProduct_gold")

# Create Sales_gold dataframe
dffactSales_gold = df.alias("df1").join(dfdimCustomer_temp.alias("df2"), (df.CustomerName == dfdimCustomer_temp.CustomerName) & (df.Email == dfdimCustomer_temp.Email), "left") \
    .join(dfdimProduct_temp.alias("df3"), (df.ItemName == dfdimProduct_temp.ItemName) & (df.ItemInfo == dfdimProduct_temp.ItemInfo), "left") \
    .select(
        col("df2.CustomerID"),
        col("df3.ItemID"),
        col("df1.OrderDate"),
        col("df1.Quantity"),
        col("df1.UnitPrice"),
        col("df1.Tax")
    ).orderBy(col("df1.OrderDate"), col("df2.CustomerID"), col("df3.ItemID"))

# Show the result
dffactSales_gold.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, 'Tables/factsales_gold')

dfUpdates = dffactSales_gold

deltaTable.alias('silver') \
.merge(
    dfUpdates.alias('updates'),
    'silver.OrderDate = updates.OrderDate AND silver.CustomerID = updates.CustomerID AND silver.ItemID = updates.ItemID'
) \
.whenMatchedUpdate(set =
    {
}
) \
.whenNotMatchedInsert(values =
    {
    "CustomerID": "updates.CustomerID",
    "ItemID": "updates.ItemID",
    "OrderDate": "updates.OrderDate",
    "Quantity": "updates.Quantity",
    "UnitPrice": "updates.UnitPrice",
    "Tax": "updates.Tax"
    }
) \
.execute()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
