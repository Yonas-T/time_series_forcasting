import pandas as pd
import numpy as np

class DataExploration:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_data_frame(self):
        return self.df

    def get_copy_of_data_frame(self):
        return self.df.copy()

    def get_column_names(self):
        return self.df.columns

    def get_data_frame_shape(self):
        return self.df.shape

    def get_data_frame_description(self):
        return self.df.describe(include=np.object)

    def get_null_values_in_data_frame(self):
        return self.df.isnull().sum()

    def get_a_column_in_data_frame(self, column_name: str):
        return self.df[column_name]

    
    
