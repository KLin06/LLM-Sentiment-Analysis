# Use a pipeline as a high-level helper
from transformers import pipeline
from PredictionMetrics import PredictionMetrics
import pandas as pd

input_file = "datasets/TD_Headlines.xlsx"

df = pd.read_excel(input_file)
#df = df.rename(columns={"Sentence": "headline", "Sentiment": "sentiment"})
df = df[(df["sentiment"] != "") & df["sentiment"].notna()]
df = df.head(1000)
headlines = df["headline"].to_list()

pipe = pipeline("text-classification", model="ProsusAI/finbert")
outputs = pipe(headlines)
preds = [output['label'] for output in outputs]
df["predicted_sentiment"] = preds
print(df)

predictionMetrics = PredictionMetrics()
predictionMetrics.set_df(df)
predictionMetrics.print_accuracy()
