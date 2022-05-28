import pandas as pd
from sklearn.model_selection import train_test_split

class SplitDataset:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.split_dataset()

    def split_dataset(self):
        feat_cols = ['DayOfWeek','Promo','StateHoliday','SchoolHoliday','Year', 'Open',
                     'Month','Day','Weekends','StoreType','Assortment','CompetitionDistance',
                     'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2',
                     'Promo2SinceWeek','Promo2SinceYear', 'PromoInterval', "CategoryInMonth"]

        # feat_cols = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 
        # 'SchoolHoliday', 'Year', 'Month', 'Day', 'CategoryInMonth', 'Weekends', 'StoreType', 'Assortment', 
        # 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 
        # 'Promo2SinceYear', 'PromoInterval']
        
        X = self.df[feat_cols]
        y = self.df["Sales"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test