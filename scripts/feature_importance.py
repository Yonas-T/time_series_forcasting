from sklearn.linear_model import LinearRegression
import pandas as pd

class FeatureImportance:

    def __init__(self, X_train, model):
        self.X_train = X_train
        self.model = model
        

    def feature_importance(self):
        if (type(self.model.steps[1][1]) == type(LinearRegression())):
            self.model = self.model.steps[1][1]
           
            p_df = pd.DataFrame()
            p_df['feature'] = self.X_train.columns.to_list()
            p_df['coff_importance'] = abs(self.model.coef_)
            
            return p_df
        
        importance = self.model.steps[1][1].feature_importances_
        feature_importance = pd.DataFrame(columns=["feature", "importance"])
        feature_importance["feature"] = self.X_train.columns.to_list()
        feature_importance["importance"] = importance
        return feature_importance