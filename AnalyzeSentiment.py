import pandas as pd
import ollama
import json
import time
from Sentiment import Sentiment

class AnalyzeSentiment():
    def __init__(self, input_file: str, output_file: str, specs_file: str, model: str, samples: int, workers: int, base_message: list):
        self.input_file = input_file
        self.output_file = output_file
        self.specs_file = specs_file
        self.base_message = base_message
        self.model = model
        self.samples = samples
        self.workers = workers
        self.df = self._get_df(input_file, samples)
        self.elapsed = None
    
    def _get_df(self, input_file: str, samples: int) -> pd.DataFrame:
        df = pd.read_csv(input_file)
        df = df.rename(columns={"Sentence": "headline", "Sentiment": "sentiment"})
        return df.head(samples)
    
    def _process_response(self, response) -> str:
        try:
            parsed = json.loads(response["message"]["content"].strip())
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}, response={response}")
            return "ERROR"

        parsed = self._normalize_prediction(parsed)
        try:
            sentiment = Sentiment(**parsed)
            return sentiment.sentiment
        except Exception as e:
            print(f"Sentiment model error: {e}, parsed={parsed}")
            return "ERROR"
        
    def _query_ollama(self, model: str, messages: str, temperature: float) -> str:
        response = ollama.chat(
            model= model,
            messages=messages,
            format="json",
            options={"temperature": temperature}
        )
        return response
    
    def _normalize_prediction(self, parsed: dict) -> dict:
        try:
            if "sentiment" in parsed and isinstance(parsed["sentiment"], str):
                s = parsed["sentiment"].strip().lower()
                if s in {"positive", "neutral", "negative"}:
                    parsed["sentiment"] = s
                else:
                    parsed["sentiment"] = "ERROR"
            else:
                parsed["sentiment"] = "ERROR"
        except Exception as e:
                print(f"Normalization error: {e}, parsed={parsed}")
                parsed["sentiment"] = "ERROR"
        return parsed
    
    def _generate_specifications(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "prompt":  json.dumps(self.base_message, indent=2),
            "elapsed_time": round(self.elapsed,2),
            "model": self.model,
            "samples": self.samples,
            "workers": self.workers
        }])
    
    def predict_sentiments(self) -> None:
        start = time.perf_counter()
        temperature = 0.0 if self.workers == 1 else 0.2
        
        for worker in range (1, self.workers + 1):
            preds = []
            for i, row in self.df.iterrows():
                headline = row["headline"]
                messages = self.base_message + [
                    {"role": "user", "content": f"Headline: {headline}"}
                ]

                try:
                    response = self._query_ollama(self.model, messages, temperature)
                    preds.append(self._process_response(response))
                except Exception as e:
                    print(f"Row {i+1} failed: {headline} -> {e}")
                    preds.append("ERROR") 
                    
            self.df["predicted_sentiment" if self.workers == 1 else f"predicted_sentiment_{worker}"] = preds
        end = time.perf_counter()
        self.elapsed = (end - start) * 1000
    
    def save(self) -> None:
        self._generate_specifications().to_csv(self.specs_file, index=False)
        self.df.to_csv(self.output_file, index = False)

