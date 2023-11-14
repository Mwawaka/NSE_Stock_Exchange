from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import pandas as pd


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, type_cast=None, to_scale=None, to_return=None, company_code=None):
        self.type_cast = type_cast
        self.to_scale = to_scale
        self.to_return = to_return
        self.company_code = company_code

    def __process(self, dataframe):
        data_scaler = StandardScaler()
        for column_name in self.type_cast:
            dataframe[column_name] = pd.to_numeric(dataframe[column_name])
        dataframe[self.to_scale] = data_scaler.fit_transform(dataframe[self.to_scale])
        return dataframe[self.to_return]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        company_df = X[X['Code'] == self.company_code].reset_index(drop=True)

        processed_df = self.__process(company_df)

        return processed_df

    # using the preprocessor


# Chooses a random company code
# random_company_code = random.choice(df.Code)
df = pd.read_csv('')
data = DataPreprocessor(
    type_cast=['Day Price', 'Day High', 'Day Low', '12m High', '12m Low'],
    to_scale=['Previous', 'Day High', 'Day Low', '12m High', '12m Low'],
    to_return=['Code', '12m High', '12m Low', 'Day Low', 'Day High', 'Previous', 'Day Price'],
    company_code='LKL')
transformed_df = data.fit_transform(df)
