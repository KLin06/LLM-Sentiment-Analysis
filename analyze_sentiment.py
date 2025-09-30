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
    "content": """You are an expert investor, and your financial success depends on correctly analyzing these sentiments.

    Task: Given a company news headline, classify the investment sentiment and output ONLY valid JSON:
    {
        "headline": "...",
        "sentiment": "positive" | "neutral" | "negative"
    }

    Deliberate reasoning (INTERNAL ONLY — do not reveal):
    1) Parse the headline: identify entity, event/action, direction/scale words, and any causes (e.g., regulation, demand).
    2) Extract features: earnings beat/miss; guidance raise/cut/withdrawal; revenue growth/decline; regulatory approval/fine/investigation; product launch/recall/delay; M&A/divestiture; layoffs/hiring; macro policy/data surprises.
    3) Map features to investor impact (buy/hold/sell). Prefer explicit results/guidance over vague phrasing.
    4) Apply decision boundaries and resolve conflicts:
       - Clearly good for shareholders → positive.
       - Clearly bad for shareholders → negative.
       - Mixed/ambiguous/weak signal → neutral.
       - If uncertain or speculative (“may”, “considering”) → neutral.
    5) Sanity checks: ignore sensational adjectives; avoid inferring beyond the headline; choose the dominant financial impact.

    Definitions:
    - "positive": Reasons an investor would be more confident to buy/hold (e.g., beats, raised guidance, strong demand, favorable regulatory decisions).
    - "negative": Reasons an investor would be less confident or consider selling/avoiding (e.g., misses, lowered/withdrawn guidance, legal issues, fines, layoffs, product failures).
    - "neutral": Information without a clear or dominant impact (e.g., scheduled announcements, leadership changes, routine updates, mixed/ambiguous signals).

    Output rules:
    - Keep all reasoning internal; do NOT include it in the final response.
    - The final response must be ONLY the JSON object shown above.
    - Do not include explanations.
    - Do not add extra fields.
    - If uncertain, always default to "neutral".
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
        "headline": "Harley-Davidson withdraws 2025 outlook due to tariff situation",
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

