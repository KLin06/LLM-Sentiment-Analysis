from PredictionMetrics import PredictionMetrics

input_file = "past prompts/42_1000_predictions.csv"
specs_file = incorrect_file = ""
prediction_metrics = PredictionMetrics(input_file, specs_file, incorrect_file)
prediction_metrics.get_confusion_matrix()