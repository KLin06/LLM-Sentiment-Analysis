import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

class PredictionMetrics():
    def __init__(self, output_file: str, specs_file: str, incorrect_file: str):
        self.output_file = output_file
        self.specs_file = specs_file
        self.incorrect_file = incorrect_file
        self.df = self._process_df(self._get_df(output_file))
        
    def _get_df(self, input_file: str) -> pd.DataFrame:
        return pd.read_csv(input_file)
    
    def _get_num_rows(self, df: pd.DataFrame) -> int:
        return len(df)

    def _get_num_correct_rows(self, df: pd.DataFrame) -> int:
        return (df["sentiment"].str.lower() == df["predicted_sentiment"].str.lower()).sum()
    
    def _get_incorrect_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["sentiment"].str.lower() != df["predicted_sentiment"].str.lower()]
    
    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["predicted_sentiment"])
        mask = (df["predicted_sentiment"] != "UNCERTAIN") & (df["predicted_sentiment"] != "ERROR")
        df = df[mask]
        return df
        
    def _save_incorrect_rows(self) -> None:
        self._get_incorrect_rows(self.df).to_csv(self.incorrect_file, index = False)
        
    def _save_specs(self) -> None:
        specs = pd.read_csv(self.specs_file)
        specs["accuracy"] = self.get_accuracy()
        spread = self.get_spread()
        predicted_spread = self.get_predicted_spread()

        specs["spread_pos"] = spread[0]
        specs["spread_neu"] = spread[1]
        specs["spread_neg"] = spread[2]

        specs["pred_pos"] = predicted_spread[0]
        specs["pred_neu"] = predicted_spread[1]
        specs["pred_neg"] = predicted_spread[2]
        
        specs.to_csv(self.specs_file, index=False)
    
    def get_f1_score(self) -> float:
        return f1_score(self.df["sentiment"], self.df["predicted_sentiment"], average = "macro")

    def get_accuracy(self) -> float:
        accuracy = self._get_num_correct_rows(self.df) / self._get_num_rows(self.df)
        return accuracy
    
    def get_spread(self) -> list:
        num_positive =  (self.df["sentiment"] == "positive").sum()
        num_neutral =  (self.df["sentiment"] == "neutral").sum()
        num_negative =  (self.df["sentiment"] == "negative").sum()
        return [num_positive, num_neutral, num_negative]
    
    def get_predicted_spread(self) -> list:
        num_positive =  (self.df["predicted_sentiment"] == "positive").sum()
        num_neutral =  (self.df["predicted_sentiment"] == "neutral").sum()
        num_negative =  (self.df["predicted_sentiment"] == "negative").sum()
        return [num_positive, num_neutral, num_negative]
    
    def get_confusion_matrix(self):
        classes = ["positive", "neutral", "negative"]
        cm = confusion_matrix(self.df["sentiment"], self.df["predicted_sentiment"], labels = classes, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
        disp.plot(cmap=plt.cm.Blues, values_format=".2f")
        plt.title('Normalized Confusion Matrix')
        plt.show()
        plt.savefig("confusion_matrix.png")
    
    def print_accuracy(self) -> None:
        print(f"Accuracy: {round(self.get_accuracy(), 2)}")
        
    def print_spread(self) -> None:
        spread = self.get_spread()
        print("***** Sample Spread *****")
        print(f"positive: {spread[0]}, neutral: {spread[1]}, negative: {spread[2]}")
    
    def print_predicted_spread(self) -> None:
        spread = self.get_predicted_spread()
        print("***** Predicted Spread *****")
        print(f"positive: {spread[0]}, neutral: {spread[1]}, negative: {spread[2]}")
    
    def save(self) -> None:
        self._save_incorrect_rows()
        self._save_specs()
    