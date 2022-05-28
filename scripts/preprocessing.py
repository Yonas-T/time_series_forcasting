import pandas as pd
import numpy as np
from sklearn import preprocessing
import logging

class Preprocessing: 
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
        pass

    def month_category(self, value):
        try:
            if (value >= 1 and int(value) < 10):
                return "Start"

            elif (value >= 10 and value < 20):
                return "Mid"
            else:
                return "End"
        except:
            logging.exception('')


    def transform_date(self, df):

        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = pd.DatetimeIndex(df['Date']).year
        df['Month'] = pd.DatetimeIndex(df['Date']).month
        df['Day'] = pd.DatetimeIndex(df['Date']).day
        df['Year'] = df['Year'].astype("int")
        df['Month'] = df['Month'].astype("category")
        df['Day'] = df['Day'].astype("category")
        df['CategoryInMonth'] = df['Day'].apply(lambda x: self.month_category(x))
        df['CategoryInMonth'] = df['CategoryInMonth'].astype("category")
        return df

    def encode_train_data(self, df):

        StateHolidayEncoder = preprocessing.LabelEncoder()
        CategoryInMonthEncoder = preprocessing.LabelEncoder()

        df['StateHoliday'] = StateHolidayEncoder.fit_transform(
            df['StateHoliday'])
        df['CategoryInMonth'] = CategoryInMonthEncoder.fit_transform(df['CategoryInMonth'])
        return df

    def encode_store_data(self, df):
        StoreTypeEncoder = preprocessing.LabelEncoder()
        AssortmentEncoder = preprocessing.LabelEncoder()
        PromoIntervalEncoder = preprocessing.LabelEncoder()

        df['StoreType'] = StoreTypeEncoder.fit_transform(df['StoreType'])
        df['Assortment'] = AssortmentEncoder.fit_transform(df['Assortment'])
        df['PromoInterval'] = PromoIntervalEncoder.fit_transform(df['PromoInterval'])

        return df

    def merge_encoded_data(self, train_encoded_data, store_encoded_data):
        return pd.merge(train_encoded_data, store_encoded_data, on="Store")

    def add_is_week_day(self, df):
        df["Weekends"] = df["DayOfWeek"].apply(lambda x: 1 if x > 5 else 0)
        df["Weekends"] = df["Weekends"].astype("category")
        return df

    def handle_outliers(self, df, col, method):

        first_quantile = df[col].quantile(0.25)
        third_quantile = df[col].quantile(0.75)
        
        low_outlier = first_quantile - ((1.5) * (third_quantile - first_quantile))
        high_outlier = third_quantile + ((1.5) * (third_quantile - first_quantile))

        if method == "mean":
            df[col] = np.where(df[col] < low_outlier,
                               df[col].mean(), df[col])
            df[col] = np.where(df[col] > high_outlier, df[col].mean(), df[col])

        elif method == "mode":
            df[col] = np.where(df[col] < low_outlier,
                               df[col].mode()[0], df[col])
            df[col] = np.where(df[col] > high_outlier,
                               df[col].mode()[0], df[col])
        else:
            df[col] = np.where(df[col] < low_outlier, low_outlier, df[col])
            df[col] = np.where(df[col] > high_outlier, high_outlier, df[col])

        return df

    def preprocess_data(self, train_df, store_df, is_test_data=False):

        train_df = self.transform_date(train_df)
        train_df = self.add_is_week_day(train_df)
        train_df = self.encode_train_data(train_df)
        store_df = self.encode_store_data(store_df)

        if (not is_test_data):
            train_df = self.handle_outliers(train_df, "Sales", '')
            train_df = self.handle_outliers(train_df, "Customers", '')

        store_df = self.handle_outliers(store_df, "CompetitionDistance", '')

        merged = self.merge_encoded_data(train_df, store_df).drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])

        return merged