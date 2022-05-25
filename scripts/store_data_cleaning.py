import pandas as pd

class StoreDataCleaning:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing_value(self, df):
        """We handled CompetitionDistance by replacing it with median"""

        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
            df['CompetitionDistance'].max())
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(
            df['Promo2SinceWeek'].max())
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(
            df['Promo2SinceWeek'].max())
        df['PromoInterval'] = df['PromoInterval'].fillna(
            df['PromoInterval'].mode()[0])
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(
            df['CompetitionOpenSinceYear'].mode()[0])
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(
            df['CompetitionOpenSinceMonth'].mode()[0])

        return df

    def convert_to_numeric(self, df):

        df["CompetitionDistance"] = df["CompetitionDistance"].astype("float")
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].astype("int")
        return df

    def convert_to_category(self, df):

        df["StoreType"] = df["StoreType"].astype("category")
        df["Assortment"] = df["Assortment"].astype("category")
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].astype(
            "category")
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].astype(
            "category")

        df["Promo2"] = df["Promo2"].astype("category")

        df["Promo2SinceYear"] = df["Promo2SinceYear"].astype("category")
        df["PromoInterval"] = df["PromoInterval"].astype("category")

        return df

    def get_cleaned_data_frame(self):
        df = self.handle_missing_value(self.df)
        df = self.convert_to_category(self.df)
        df = self.convert_to_numeric(self.df)
        return df

    