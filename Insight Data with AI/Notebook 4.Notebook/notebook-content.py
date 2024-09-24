# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "31700178-e2d1-42c9-ac93-5b76e04ef172",
# META       "default_lakehouse_name": "AI_Fabric_Lakehouse",
# META       "default_lakehouse_workspace_id": "0b00d2c0-b0ef-4be3-93e9-b46cf35f9624",
# META       "known_lakehouses": [
# META         {
# META           "id": "31700178-e2d1-42c9-ac93-5b76e04ef172"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import pandas as pd
from tqdm.auto import tqdm
base = "https://synapseaisolutionsa.blob.core.windows.net/public/AdventureWorks"

    # load list of tables
df_tables = pd.read_csv(f"{base}/adventureworks.csv", names=["table"])

for table in (pbar := tqdm(df_tables['table'].values)):
    pbar.set_description(f"Uploading {table} to lakehouse")
    # download
    df = pd.read_parquet(f"{base}/{table}.parquet")

# save as lakehouse table
    spark.createDataFrame(df).write.mode('overwrite').saveAsTable(table)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
