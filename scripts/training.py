import mlflow
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from data_transforming import DataTransforming
import pandas as pd

class Training:
    def __init__(self, X_train, y_train, model_name, df):
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        self.df = df

    def train(self, regressor=RandomForestRegressor(n_jobs=-1, max_depth=15, n_estimators=15)):
        feat_cols = ['DayOfWeek','Promo','StateHoliday','SchoolHoliday','Year', 'Open',
                     'Month','Day','Weekends','StoreType','Assortment','CompetitionDistance',
                     'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2',
                     'Promo2SinceWeek','Promo2SinceYear', 'PromoInterval', "CategoryInMonth"]

        pre = DataTransforming(self.df[feat_cols]).preprocessing()
        
        pipeline = Pipeline(steps=[('preprocessor', pre),
                                   ('regressor', regressor)])
        
        mlflow.set_experiment('Rossman_' + self.model_name)
        mlflow.sklearn.autolog()
        with mlflow.start_run(run_name="Baseline"):
            model = pipeline.fit(self.X_train, self.y_train)
        return pipeline, model
