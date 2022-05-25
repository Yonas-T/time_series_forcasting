import pandas as pd
import logging

class TrainDataCleaning:

    logging.basicConfig(level=logging.INFO)

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def convert_to_number(self, df):
        df["Customers"] = df["Customers"].astype("int")
        df["Sales"] = df["Sales"].astype("int")
        return df

    def convert_to_datatime(self, df):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except :
            logging.exception('')

    def convert_to_category(self, df):

        df["Open"] = df["Open"].astype("category")
        df["Promo"] = df["Promo"].astype("category")
        df["StateHoliday"] = df["StateHoliday"].astype("category")
        df["SchoolHoliday"] = df["SchoolHoliday"].astype("category")
        df['StateHoliday'] = df['StateHoliday'].astype(
            "str").astype("category")
        return df

    def get_cleaned_data_frame(self):
        df = self.convert_to_number(self.df)
        df = self.convert_to_datatime(self.df)
        df = self.convert_to_category(self.df)
        return df