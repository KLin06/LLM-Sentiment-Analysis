# This is the main script

from AnalyzeSentiment import AnalyzeSentiment
from IterationCounter import IterationCounter
from AggregatePredictions import AggregatePredictions
from PredictionMetrics import PredictionMetrics

# models: llama3:8b, llama3.2:3b, mistral:7b
model = "llama3:8b"
samples = 1000
# try to use 1 or 5 workers 
workers = 1

iteration = IterationCounter()
iteration_val = iteration.get()
input_file = "datasets/dataset.csv"
output_file = f"past prompts/{iteration_val}_{samples}_predictions.csv"
specs_file = f"past prompts/{iteration_val}_{samples}_specifications.csv"
incorrect_file = f"past prompts/{iteration_val}_{samples}_incorrect_predictions.csv"

base_message = [
{
    "role": "system",
    "content": """You are an expert investor.

    Task: Given a company news headline:
    Step 1: Internally translate the headline into a weather forecast.
       • Sunny = positive
       • Cloudy = neutral
       • Stormy = negative
    Step 2: Map the weather outcome back into investment sentiment.
    Step 3: Output ONLY valid JSON:
    {
        "sentiment": "positive" | "neutral" | "negative"
    }

    Rules:
    - The weather metaphor is part of your internal reasoning. Do not show it in the output.
    - The final response must ONLY be the JSON object.
    - Do not add explanations.
    - Do not add extra fields.
    - Sentiment must be exactly one of: positive, neutral, negative.
    """
},
{
    "role": "user",
    "content": "Headline: Harley-Davidson withdraws 2025 outlook due to tariff situation"
},
{
    "role": "assistant",
    "content": """{
        "sentiment": "negative"
    }"""
}


]

analyze_sentiment = AnalyzeSentiment(input_file, output_file, specs_file, model, samples, workers, base_message)
analyze_sentiment.predict_sentiments()
analyze_sentiment.save()

# combine the different workers
if workers > 1:
    aggregate_predictions = AggregatePredictions(output_file, workers)
    aggregate_predictions.save_aggregated_predictions()

# save the metrics
prediction_metrics = PredictionMetrics(output_file, specs_file, incorrect_file)
prediction_metrics.print_accuracy()
prediction_metrics.print_spread()
prediction_metrics.print_predicted_spread()
prediction_metrics.save()

# update the interation data
iteration.increment_save()

