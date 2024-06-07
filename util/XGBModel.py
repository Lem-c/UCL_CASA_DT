import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


class XGB:

    def __init__(self, df_):
        self.df = df_

    def train(self, X_=None, y_='ww_value'):
        # Extract features (X) and target (y)
        if X_ is None:
            X_ = ['vir_value']
        X = self.df[X_]
        y = self.df[y_]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2024)

        # Initialize the XGBRegressor model
        model = XGBRegressor()

        # Train the model
        model.fit(X_train, y_train)

        # Predict using the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
        return model
