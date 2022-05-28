from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import logging
class Testing:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        logging.basicConfig(level=logging.INFO)
        

    def test(self):
        
        predictions = self.model.predict(self.X_test)
        score_2 = r2_score(self.y_test, predictions)
        loss = mean_absolute_error(predictions, self.y_test)

        logging.info('R2 score: {score_2:.3f}')
        logging.info('Mean Absolute Error: {loss:.3f}')

        result_df = self.X_test.copy()
        result_df["Prediction"] = predictions
        result_df["Actual"] = self.y_test
        result_aggregate = result_df.groupby("Day").agg({"Prediction": "mean", "Actual":"mean"})
        
        return score_2, loss, result_aggregate