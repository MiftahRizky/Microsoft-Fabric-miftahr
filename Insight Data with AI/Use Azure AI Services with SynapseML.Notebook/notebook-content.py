# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "827d83a9-bdfe-4565-bdc0-4f6f07bbceb5",
# META       "default_lakehouse_name": "Use_case5",
# META       "default_lakehouse_workspace_id": "0b00d2c0-b0ef-4be3-93e9-b46cf35f9624",
# META       "known_lakehouses": [
# META         {
# META           "id": "827d83a9-bdfe-4565-bdc0-4f6f07bbceb5"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

from pyspark.sql.functions import udf, col
from synapse.ml.io.http import HTTPTransformer, http_udf
from requests import Request
from pyspark.sql.functions import lit
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
import os

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import SparkSession
from synapse.ml.core.platform import *

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.cognitive import *

# A general Azure AI services key for Text Analytics, Vision and Document Intelligence (or use separate keys that belong to each service)
service_key = "5064f018ad05490db8872a1eb8a6e662" # Replace <YOUR-KEY-VALUE> with your Azure AI service key, check prerequisites for more details
service_loc = "eastus2"

# A Bing Search v7 subscription key
bing_search_key =  "5064f018ad05490db8872a1eb8a6e662" # Replace <YOUR-KEY-VALUE> with your Bing v7 subscription key, check prerequisites for more details

# An Anomaly Detector subscription key
anomaly_key = "5064f018ad05490db8872a1eb8a6e662" # Replace <YOUR-KEY-VALUE> with your anomaly service key, check prerequisites for more details
anomaly_loc = "eastus2"

# A Translator subscription key
translator_key = "5064f018ad05490db8872a1eb8a6e662" # Replace <YOUR-KEY-VALUE> with your translator service key, check prerequisites for more details
translator_loc = "eastus2"

# An Azure search key
search_key = "5064f018ad05490db8872a1eb8a6e662" # Replace <YOUR-KEY-VALUE> with your search key, check prerequisites for more details

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create a dataframe that's tied to it's column names
df = spark.createDataFrame(
    [
        ("I am so happy today, its sunny!", "en-US"),
        ("I am frustrated by this rush hour traffic", "en-US"),
        ("The cognitive services on spark aint bad", "en-US"),
    ],
    ["text", "language"],
)

# Run the Text Analytics service with options
sentiment = (
    TextSentiment()
    .setTextCol("text")
    .setLocation(service_loc)
    .setSubscriptionKey(service_key)
    .setOutputCol("sentiment")
    .setErrorCol("error")
    .setLanguageCol("language")
)

# Show the results of your text query in a table format
display(
    sentiment.transform(df).select(
        "text", col("sentiment.document.sentiment").alias("sentiment")
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.createDataFrame(
    [
        ("20mg of ibuprofen twice a day",),
        ("1tsp of Tylenol every 4 hours",),
        ("6-drops of Vitamin B-12 every evening",),
    ],
    ["text"],
)

healthcare = (
    AnalyzeHealthText()
    .setSubscriptionKey(service_key)
    .setLocation(service_loc)
    .setLanguage("en")
    .setOutputCol("response")
)

display(healthcare.transform(df))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, flatten
# Create a dataframe including sentences you want to translate
df = spark.createDataFrame(
    [(["Hello, what is your name?", "Bye"],)],
    [
        "text",
    ],
)

# Run the Translator service with options
translate = (
    Translate()
    .setSubscriptionKey(translator_key)
    .setLocation(translator_loc)
    .setTextCol("text")
    .setToLanguage(["zh-Hans"])
    .setOutputCol("translation")
)

# Show the results of the translation.
display(
    translate.transform(df)
    .withColumn("translation", flatten(col("translation.translations")))
    .withColumn("translation", col("translation.text"))
    .select("translation")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, explode
# Create a dataframe containing the source files
imageDf = spark.createDataFrame(
    [
        (
            "https://mmlspark.blob.core.windows.net/datasets/FormRecognizer/business_card.jpg",
        )
    ],
    [
        "source",
    ],
)

# Run the Form Recognizer service
analyzeBusinessCards = (
    AnalyzeBusinessCards()
    .setSubscriptionKey(service_key)
    .setLocation(service_loc)
    .setImageUrlCol("source")
    .setOutputCol("businessCards")
)

# Show the results of recognition.
display(
    analyzeBusinessCards.transform(imageDf)
    .withColumn(
        "documents", explode(col("businessCards.analyzeResult.documentResults.fields"))
    )
    .select("source", "documents")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create a dataframe with the image URLs
base_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/"
df = spark.createDataFrame(
    [
        (base_url + "objects.jpg",),
        (base_url + "dog.jpg",),
        (base_url + "house.jpg",),
    ],
    [
        "image",
    ],
)

# Run the Computer Vision service. Analyze Image extracts information from/about the images.
analysis = (
    AnalyzeImage()
    .setLocation(service_loc)
    .setSubscriptionKey(service_key)
    .setVisualFeatures(
        ["Categories", "Color", "Description", "Faces", "Objects", "Tags"]
    )
    .setOutputCol("analysis_results")
    .setImageUrlCol("image")
    .setErrorCol("error")
)

# Show the results of what you wanted to pull out of the images.
display(analysis.transform(df).select("image", "analysis_results.description.tags"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Number of images Bing will return per query
imgsPerBatch = 10
# A list of offsets, used to page into the search results
offsets = [(i * imgsPerBatch,) for i in range(100)]
# Since web content is our data, we create a dataframe with options on that data: offsets
bingParameters = spark.createDataFrame(offsets, ["offset"])

# Run the Bing Image Search service with our text query
bingSearch = (
    BingImageSearch()
    .setSubscriptionKey(bing_search_key)
    .setOffsetCol("offset")
    .setQuery("Martin Luther King Jr. quotes")
    .setCount(imgsPerBatch)
    .setOutputCol("images")
)

# Transformer that extracts and flattens the richly structured output of Bing Image Search into a simple URL column
getUrls = BingImageSearch.getUrlTransformer("images", "url")

# This displays the full results returned, uncomment to use
# display(bingSearch.transform(bingParameters))

# Since we have two services, they are put into a pipeline
pipeline = PipelineModel(stages=[bingSearch, getUrls])

# Show the results of your search: image URLs
display(pipeline.transform(bingParameters))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create a dataframe with our audio URLs, tied to the column called "url"
df = spark.createDataFrame(
    [("https://mmlspark.blob.core.windows.net/datasets/Speech/audio2.wav",)], ["url"]
)

# Run the Speech-to-text service to translate the audio into text
speech_to_text = (
    SpeechToTextSDK()
    .setSubscriptionKey(service_key)
    .setLocation(service_loc)
    .setOutputCol("text")
    .setAudioDataCol("url")
    .setLanguage("en-US")
    .setProfanity("Masked")
)

# Show the results of the translation
display(speech_to_text.transform(df).select("url", "text.DisplayText"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from synapse.ml.cognitive import TextToSpeech

fs = ""
if running_on_databricks():
    fs = "dbfs:"
elif running_on_synapse_internal():
    fs = "Files"

# Create a dataframe with text and an output file location
df = spark.createDataFrame(
    [
        (
            "Reading out loud is fun! Check out aka.ms/spark for more information",
            fs + "/output.mp3",
        )
    ],
    ["text", "output_file"],
)

tts = (
    TextToSpeech()
    .setSubscriptionKey(service_key)
    .setTextCol("text")
    .setLocation(service_loc)
    .setVoiceName("en-US-JennyNeural")
    .setOutputFileCol("output_file")
)

# Check to make sure there were no errors during audio creation
display(tts.transform(df))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create a dataframe with the point data that Anomaly Detector requires
df = spark.createDataFrame(
    [
        ("1972-01-01T00:00:00Z", 826.0),
        ("1972-02-01T00:00:00Z", 799.0),
        ("1972-03-01T00:00:00Z", 890.0),
        ("1972-04-01T00:00:00Z", 900.0),
        ("1972-05-01T00:00:00Z", 766.0),
        ("1972-06-01T00:00:00Z", 805.0),
        ("1972-07-01T00:00:00Z", 821.0),
        ("1972-08-01T00:00:00Z", 20000.0),
        ("1972-09-01T00:00:00Z", 883.0),
        ("1972-10-01T00:00:00Z", 898.0),
        ("1972-11-01T00:00:00Z", 957.0),
        ("1972-12-01T00:00:00Z", 924.0),
        ("1973-01-01T00:00:00Z", 881.0),
        ("1973-02-01T00:00:00Z", 837.0),
        ("1973-03-01T00:00:00Z", 9000.0),
    ],
    ["timestamp", "value"],
).withColumn("group", lit("series1"))

# Run the Anomaly Detector service to look for irregular data
anamoly_detector = (
    SimpleDetectAnomalies()
    .setSubscriptionKey(anomaly_key)
    .setLocation(anomaly_loc)
    .setTimestampCol("timestamp")
    .setValueCol("value")
    .setOutputCol("anomalies")
    .setGroupbyCol("group")
    .setGranularity("monthly")
)

# Show the full results of the analysis with the anomalies marked as "True"
display(
    anamoly_detector.transform(df).select("timestamp", "value", "anomalies.isAnomaly")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Use any requests from the python requests library
def world_bank_request(country):
    return Request(
        "GET", "http://api.worldbank.org/v2/country/{}?format=json".format(country)
    )


# Create a dataframe with specifies which countries we want data on
df = spark.createDataFrame([("br",), ("usa",)], ["country"]).withColumn(
    "request", http_udf(world_bank_request)(col("country"))
)

# Much faster for big data because of the concurrency :)
client = (
    HTTPTransformer().setConcurrency(3).setInputCol("request").setOutputCol("response")
)

# Get the body of the response


def get_response_body(resp):
    return resp.entity.content.decode()


# Show the details of the country data returned
display(
    client.transform(df).select(
        "country", udf(get_response_body)(col("response")).alias("response")
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
