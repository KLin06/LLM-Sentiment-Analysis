import pandas as pd
input_file = "past prompts/35_1000_predictions.csv"
df = pd.read_csv(input_file)
df = df.rename (columns = {"predicted_sentiment": "predicted_sentiment_1"})
df.to_csv(input_file, index = False)