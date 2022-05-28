from lstm_prediction import LstmPrediction
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class ForcastSales:
    def __init__(self, batch_size, scaled_df, model, data_agg) -> None:
        self.batch_size = batch_size
        self.scaled_df = scaled_df
        self.model = model
        self.data_agg = data_agg
        pass


    def forcast_next_one_sale(self, sales):
        data_feat = None
        WINDOW_SIZE = 49
        try:
            data_feat = sales[["Sales", "Date"]]
            
            if (data_feat.shape[0] < 49):
                print("To make prediction, we need atleast data of 49 dates")
                return
            scaled_dff, scaler_obj = LstmPrediction(WINDOW_SIZE, self.batch_file, self.scaled_df).add_scaled_sales(data_feat)
            data_feat["Sales"].isna().any().sum()
            SIZE = len(self.data_agg["Sales"])
            
            
            series = scaled_dff["scaled_sales"].values[:, np.newaxis]

            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True) 
            ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
            ds = ds.batch(SIZE).prefetch(1)
            
            forecast = self.model.predict(ds)
            Results = list(forecast.reshape(1, forecast.shape[0] * forecast.shape[1])[0].copy())

            Results1 = scaler_obj.inverse_transform(forecast.reshape(-1,1))
            Results1 = list(Results1.reshape(1, Results1.shape[0] * Results1.shape[1])[0])
            
            return  Results1, Results
            
        except KeyError as e:
            print(e)
            return False

    def forcast_next_sales(self, model, sales, daysToForcast=1):
        forcasts = []
        scaled_forcasts = []
        dates = []
        
        new_sales_df = sales.copy()
        while len(forcasts) < daysToForcast:
            forcast, scaled_forcast = self.forcast_next_one_sale(model, new_sales_df)
            forcasts += forcast
            
            scaled_forcasts += scaled_forcast
            size=len(new_sales_df["Sales"])
            
            truncated_sales = new_sales_df.tail(size - len(scaled_forcast))
                    
            new_sales = truncated_sales['Sales'].to_list() + scaled_forcast
            next_dates = []
            
            for i in range(len(scaled_forcast)):
                next_date = new_sales_df["Date"].to_list()[-1] + timedelta(days=1)
                next_dates.append(next_date)     
            
            new_dates = truncated_sales['Date'].to_list() + next_dates
            new_sales_df = pd.DataFrame()
            new_sales_df["Date"] = new_dates
            new_sales_df["Sales"] = new_sales
        
        res_df = pd.DataFrame()
        res_df["Date"] = new_dates
        res_df["forcasts"] = forcasts
        
        return res_df

    def get_prediction_data_frame(self):
        self.scaled_df["Date"] = self.scaled_df.index
        self.scaled_df["Date"] = self.scaled_df["Date"].astype("datetime64[ns]")
        res_df = self.forcast_next_sales(self.model, self.scaled_df.head(49), 49)
        return res_df