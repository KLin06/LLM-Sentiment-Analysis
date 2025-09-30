from collections import Counter
import pandas as pd

class AggregatePredictions():
    def __init__(self, input_file: str, workers: int, output_file: str = None):
        self.input_file = input_file
        
        if output_file == None:
            self.output_file = input_file
        else:
            self.output_file = output_file
        
        self.workers = workers
        self.df = self._process_df(self._get_df(input_file))
        
    def _get_df(self, input_file: str) -> pd.DataFrame:
        return pd.read_csv(input_file)
        
    def _process_df (self, df: pd.DataFrame) -> pd.DataFrame:
        preds = []
        for _, row in df.iterrows():
            predictions = []
            
            for worker in range(1, self.workers + 1):
                predictions.append(row[f"predicted_sentiment_{worker}"])
                
            most_common_prediction = self._get_most_common_prediction(predictions)
            preds.append(most_common_prediction)
        
        df["predicted_sentiment"] = preds
        return df
        
    def _get_most_common_prediction(self, predictions: list) -> str: 
        predictions = [prediction for prediction in predictions if prediction not in ["ERROR", None]]
        if not predictions:
            return "ERROR"
        
        counts = Counter(predictions)
        most_common = counts.most_common()
        
        if len(most_common) == 1:
            return most_common[0][0]
        if most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        if most_common[0][1] == most_common[1][1] and (most_common[0][0] == "neutral" or most_common[1][0] == "neutral"):
            return "neutral"
        return "UNCERTAIN"
    
    def save_aggregated_predictions(self) -> None:
        self.df.to_csv(self.output_file, index = False)