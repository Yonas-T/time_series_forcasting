import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class DataTransforming:
    def __init__(self, preprocessed_data: pd.DataFrame):
        self.preprocessed_data = preprocessed_data
      

    def preprocessing(self):
        numric_cols = ["CompetitionDistance", "Promo2SinceWeek", "Year"]
        categorical_cols = self.preprocessed_data.copy(deep=True).drop(columns=numric_cols, axis=1, inplace=False).columns.to_list()
        
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler()),
                                                ('imputer', SimpleImputer(strategy='mean'))])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                    ('encoder', OrdinalEncoder())])
        
        preprocessing = ColumnTransformer(
            transformers=[('numric', numeric_transformer, numric_cols),
                            ('category', categorical_transformer, categorical_cols)])
        return preprocessing