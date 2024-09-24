# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "7046a21c-2e4a-43f5-9e63-7eaf6e12f780",
# META       "default_lakehouse_name": "DocumentIntelligence",
# META       "default_lakehouse_workspace_id": "0b00d2c0-b0ef-4be3-93e9-b46cf35f9624",
# META       "known_lakehouses": [
# META         {
# META           "id": "7046a21c-2e4a-43f5-9e63-7eaf6e12f780"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Azure AI Search
AI_SEARCH_NAME = "mysearchservice4421"
AI_SEARCH_INDEX_NAME = "rag-demo-index"
AI_SEARCH_API_KEY = "33165e17b5724ebea3a02126d1b01a72"

# Azure AI Services
AI_SERVICES_KEY = "5064f018ad05490db8872a1eb8a6e662"
AI_SERVICES_LOCATION = "eastus2"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import requests
import os

url = "https://github.com/Azure-Samples/azure-openai-rag-workshop/raw/main/data/support.pdf"
response = requests.get(url)

# Specify your path here
path = "/lakehouse/default/Files/"

# Ensure the directory exists
os.makedirs(path, exist_ok=True)

# Write the content to a file in the specified path
filename = url.rsplit("/")[-1]
with open(os.path.join(path, filename), "wb") as f:
    f.write(response.content)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
document_path = f"Files/{filename}"
df = spark.read.format("binaryFile").load(document_path).select("_metadata.file_name", "content").limit(10).cache()
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.services import AnalyzeDocument
from pyspark.sql.functions import col

analyze_document = (
    AnalyzeDocument()
    .setPrebuiltModelId("prebuilt-layout")
    .setSubscriptionKey("5064f018ad05490db8872a1eb8a6e662")
    .setLocation("eastus2")
    .setImageBytesCol("content")
    .setOutputCol("result")
)

analyzed_df = (
    analyze_document.transform(df)
    .withColumn("output_content", col("result.analyzeResult.content"))
    .withColumn("paragraphs", col("result.analyzeResult.paragraphs"))
).cache()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

analyzed_df = analyzed_df.drop("content")
display(analyzed_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.featurize.text import PageSplitter

ps = (
    PageSplitter()
    .setInputCol("output_content")
    .setMaximumPageLength(4000)
    .setMinimumPageLength(3000)
    .setOutputCol("chunks")
)

splitted_df = ps.transform(analyzed_df)
display(splitted_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import posexplode, col, concat

# Each "chunks" column contains the chunks for a single document in an array
# The posexplode function will separate each chunk into its own row
exploded_df = splitted_df.select("file_name", posexplode(col("chunks")).alias("chunk_index", "chunk"))

# Add a unique identifier for each chunk
exploded_df = exploded_df.withColumn("unique_id", concat(exploded_df.file_name, exploded_df.chunk_index))

display(exploded_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.services import OpenAIEmbedding

embedding = (
    OpenAIEmbedding()
    .setDeploymentName("text-embedding-ada-002")
    .setTextCol("chunk")
    .setErrorCol("error")
    .setOutputCol("embeddings")
)

df_embeddings = embedding.transform(exploded_df)

display(df_embeddings)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import requests
import json

# Length of the embedding vector (OpenAI ada-002 generates embeddings of length 1536)
EMBEDDING_LENGTH = 1536

# Create index for AI Search with fields id, content, and contentVector
# Note the datatypes for each field below
url = f"https://{AI_SEARCH_NAME}.search.windows.net/indexes/{AI_SEARCH_INDEX_NAME}?api-version=2023-11-01"
payload = json.dumps(
    {
        "name": AI_SEARCH_INDEX_NAME,
        "fields": [
            # Unique identifier for each document
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "filterable": True,
            },
            # Text content of the document
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
            },
            # Vector embedding of the text content
            {
                "name": "contentVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": True,
                "dimensions": EMBEDDING_LENGTH,
                "vectorSearchProfile": "vectorConfig",
            },
        ],
        "vectorSearch": {
            "algorithms": [{"name": "hnswConfig", "kind": "hnsw", "hnswParameters": {"metric": "cosine"}}],
            "profiles": [{"name": "vectorConfig", "algorithm": "hnswConfig"}],
        },
    }
)
headers = {"Content-Type": "application/json", "api-key": AI_SEARCH_API_KEY}

response = requests.request("PUT", url, headers=headers, data=payload)
if response.status_code == 201:
    print("Index created!")
elif response.status_code == 204:
    print("Index updated!")
else:
    print(f"HTTP request failed with status code {response.status_code}")
    print(f"HTTP response body: {response.text}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import re

from pyspark.sql.functions import monotonically_increasing_id

def insert_into_index(documents):
    """Uploads a list of 'documents' to Azure AI Search index."""
    url = f"https://{AI_SEARCH_NAME}.search.windows.net/indexes/{AI_SEARCH_INDEX_NAME}/docs/index?api-version=2023-11-01"
    payload = json.dumps({"value": documents})
    headers = {
        "Content-Type": "application/json",
        "api-key": AI_SEARCH_API_KEY,
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200 or response.status_code == 201:
        return "Success"
    else:
        return f"Failure: {response.text}"
def make_safe_id(row_id: str):
    """Strips disallowed characters from row id for use as Azure AI search document ID."""
    return re.sub("[^0-9a-zA-Z_-]", "_", row_id)

def upload_rows(rows):
    """Uploads the rows in a Spark dataframe to Azure AI Search.
    Limits uploads to 1000 rows at a time due to Azure AI Search API limits.
    """
    BATCH_SIZE = 1000
    rows = list(rows)
    for i in range(0, len(rows), BATCH_SIZE):
        row_batch = rows[i : i + BATCH_SIZE]
        documents = []
        for row in rows:
            documents.append(
                {
                    "id": make_safe_id(row["unique_id"]),
                    "content": row["chunk"],
                    "contentVector": row["embeddings"].tolist(),
                    "@search.action": "upload",
                },
            )
        status = insert_into_index(documents)
        yield [row_batch[0]["row_index"], row_batch[-1]["row_index"], status]

# Add ID to help track what rows were successfully uploaded
df_embeddings = df_embeddings.withColumn("row_index", monotonically_increasing_id())

# Run upload_batch on partitions of the dataframe
res = df_embeddings.rdd.mapPartitions(upload_rows)
display(res.toDF(["start_index", "end_index", "insertion_status"]))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Azure AI Search
AI_SEARCH_NAME = 'mysearchservice4421'
AI_SEARCH_INDEX_NAME = 'rag-demo-index'
AI_SEARCH_API_KEY = '0c01891857a74de8884cdd6355abf3e5'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def gen_question_embedding(user_question):
    """Generates embedding for user_question using SynapseML."""
    from synapse.ml.services import OpenAIEmbedding
    df_ques = spark.createDataFrame([(user_question, 1)], ["questions", "dummy"])
    embedding = (
        OpenAIEmbedding()
        .setDeploymentName('text-embedding-ada-002')
        .setTextCol("questions")
        .setErrorCol("errorQ")
        .setOutputCol("embeddings")
    )
    df_ques_embeddings = embedding.transform(df_ques)
    row = df_ques_embeddings.collect()[0]
    question_embedding = row.embeddings.tolist()
    return question_embedding

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import json 
import requests

def retrieve_top_chunks(k, question, question_embedding):
    """Retrieve the top K entries from Azure AI Search using hybrid search."""
    url = f"https://{AI_SEARCH_NAME}.search.windows.net/indexes/{AI_SEARCH_INDEX_NAME}/docs/search?api-version=2023-11-01"
    payload = json.dumps({
        "search": question,
        "top": k,
        "vectorQueries": [
            {
                "vector": question_embedding,
                "k": k,
                "fields": "contentVector",
                "kind": "vector"
            }
        ]
    })
    headers = {
        "Content-Type": "application/json",
        "api-key": AI_SEARCH_API_KEY,
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    output = json.loads(response.text)
    return output

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def get_context(user_question, retrieved_k = 5):
    # Generate embeddings for the question
    question_embedding = gen_question_embedding(user_question)
    # Retrieve the top K entries
    output = retrieve_top_chunks(retrieved_k, user_question, question_embedding)
    # concatenate the content of the retrieved documents
    context = [chunk["content"] for chunk in output["value"]]
    return context

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import Row
from synapse.ml.services.openai import OpenAIChatCompletion


def make_message(role, content):
    return Row(role=role, content=content, name=role)

def get_response(user_question):
    context = get_context(user_question)
    # Write a prompt with context and user_question as variables 
    prompt = f"""
    context: {context}
    Answer the question based on the context above.
    If the information to answer the question is not present in the given context then reply "I don't know".
    """
    chat_df = spark.createDataFrame(
        [
            (
                [
                    make_message(
                        "system", prompt
                    ),
                    make_message("user", user_question),
                ],
            ),
        ]
    ).toDF("messages")
    chat_completion = (
        OpenAIChatCompletion()
        .setDeploymentName("gpt-35-turbo-16k") # deploymentName could be one of {gpt-35-turbo, gpt-35-turbo-16k}
        .setMessagesCol("messages")
        .setErrorCol("error")
        .setOutputCol("chat_completions")
    )
    result_df = chat_completion.transform(chat_df).select("chat_completions.choices.message.content")
    result = []
    for row in result_df.collect():
        content_string = ' '.join(row['content'])
        result.append(content_string)
    # Join the list into a single string
    result = ' '.join(result)      
    return result

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

user_question = "how do i make a booking?"
response = get_response(user_question)
print(response)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
