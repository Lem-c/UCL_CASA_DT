import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from sklearn.preprocessing import StandardScaler


def plot_conditional_pred(y, pred, bins=(10, 20), ax=None, title=None):
    """Plot colour-map of predictions as a function of True values

    Args:
        y: array-like
            True values
        pred: array-like
            predictions (same length as y)
        bins: 2-tuple/2-list
            binning values for x (true values) and y (preds) axes
        ax: matplotlib axe
            optional axis to populate
        title:
            optional title
    Return:
        corresponding plt.axe object"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = Axis.get_figure(ax)
    ranges = [[np.min(y), np.max(y)]] * 2
    h, bins_y, bins_pred = np.histogram2d(y, pred, bins=bins, range=ranges)
    h_norm = (h / h.sum(1, keepdims=True)).T

    ax.scatter(y, pred, s=0.2, alpha=0.1, color='red')
    ax.plot(*ranges,
            c='red', label='$\hat{y}=y$', alpha=0.5)
    im = ax.imshow(h_norm,
                   extent=np.array(ranges).flatten(),
                   cmap=plt.cm.Reds, origin='lower', interpolation='nearest')

    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel(f'Proportion of estimates', fontsize=14, rotation=270, labelpad=20)

    ax.set_xlabel('True values', fontsize=14)
    ax.set_ylabel(f'Predictions', fontsize=14)
    ax.set_title(title)
    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    return ax


class XGB:

    def __init__(self, df_train, df_test, df_main=None,
                 y_scale='log10',
                 log10_variables=None,
                 input_offset=0., target_offset=0.):

        if log10_variables is None:
            log10_variables = ['sars_cov2_gc_l_mean', 'control_gc_l_mean', 'suspended_solids_mg_l',
                               'ammonia_mg_l', 'ophosph_mg_l']
        self.df_train = df_train
        self.df_test = df_test
        self.df = df_main

        # Prepare the data
        self.all_variables = ['sars_cov2_gc_l_mean', 'suspended_solids_mg_l', 'ammonia_mg_l', 'ophosph_mg_l',
                              'sample_ph_pre_ansis', 'raw_ansis_sars_repc_std', 'grab_compo_boo', 'sars_below_lod',
                              'sars_below_loq', 'reception_delay', 'catchment_population_ons_mid_2019',
                              'catchment_area', 'catch_in_cis_prop', 'catch_cis_population',
                              'ratio_over_50', 'Ethnicity_Proportion', 'imd_score']

        # Define target variable
        self.target_variable = 'infection_rate'

        self.X = self.df[self.all_variables]
        self.y = self.df[self.target_variable]

        self.y_scale = y_scale
        self.log10_variables = log10_variables
        self.scaler = StandardScaler()

        if self.y_scale == 'linear':
            self.y_transform = lambda x: x + target_offset
            self.inverse_transform_y = lambda x: x - target_offset
        elif self.y_scale == 'log':
            self.y_transform = lambda x: np.log(x + target_offset)
            self.inverse_transform_y = lambda x: np.exp(x) - target_offset
        elif self.y_scale == 'log10':
            self.y_transform = lambda x: np.log10(x + target_offset)
            self.inverse_transform_y = lambda x: 10 ** x - target_offset
        else:
            raise ValueError('y_scale must be linear or log')

        print("Mean values:\n", self.df['imd_score'].mean())
        print("\nMedian values:\n", self.df['imd_score'].median())
        print("\nMin values:\n", self.df['imd_score'].min())

        self.inverse_transform_x = dict()
        self.input_offset = input_offset
        self.x_transform = dict()

    def sep_data_process(self, remove_percent):
        """
        Method contains data generation
        :param remove_percent:
        :return:
        """

    def xgb_regression(self):
        """
        Method realize XGBoost.
        - hyper-parameter
        - lgb
        - ML
        :return:
        """

    def log10_transform(self):
        x = self.X.copy()
        # Log transfo

        for var in self.all_variables:
            if var in self.log10_variables:
                x[var] = np.log10(x[var] + self.input_offset)
                self.x_transform[var] = lambda x: np.log10(x + self.input_offset)
                self.inverse_transform_x[var] = lambda y: 10 ** y - self.input_offset
            else:
                self.x_transform[var] = lambda x: x
                self.inverse_transform_x[var] = lambda y: y

        y = self.y_transform(self.y)
        return x, y

    def random_split(self, test_size=0.25, random_state=0, standardise=False):
        x, y = self.log10_transform()

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size,
                                                            shuffle=True,
                                                            random_state=random_state)
        if standardise:
            x_train.loc[:, :] = self.scaler.fit_transform(x_train)
            x_test.loc[:, :] = self.scaler.transform(x_test)
        self.X = pd.concat([x_train, x_test])
        self.y = pd.concat([y_train, y_test])
        return x_train, x_test, y_train, y_test

    def train_xgb(self, is_date_split=False, is_spatial_split=False):

        if is_date_split:
            X_all, y_all = self.log10_transform()

            X_all = pd.concat([X_all, y_all], axis=1)
            df_temp = X_all
            # Convert the date column to datetime format
            df_temp['date'] = pd.to_datetime(self.df['date'])

            # Sort the dataset by date
            data = df_temp.sort_values(by='date')

            # Calculate the 80% point
            split_point = int(len(data) * 0.8)

            # Split the dataset into training and testing sets
            train_set = data.iloc[:split_point]
            test_set = data.iloc[split_point:]

            X = train_set[self.all_variables]
            y = train_set[self.target_variable]

            X_test = test_set[self.all_variables]
            y_test = test_set[self.target_variable]

            # Handle any missing values by filling with the mean of the column
            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)
            X_test.fillna(X_test.mean(), inplace=True)
            y_test.fillna(y_test.mean(), inplace=True)

        if is_spatial_split:

            X_all, y_all = self.log10_transform()
            # X_all, y_all = self.X, self.y

            group_kfold = GroupKFold(n_splits=5)
            groups = self.df['CIS20CD']
            # Performing the split
            splits = list(group_kfold.split(X_all, y_all, groups=groups))
            X_all = pd.concat([X_all, y_all], axis=1)
            # Display the indices for one of the splits to demonstrate
            train_indices, test_indices = splits[0]
            train_set = X_all.iloc[train_indices]
            test_set = X_all.iloc[test_indices]

            X = train_set[self.all_variables]
            y = train_set[self.target_variable]

            X_test = test_set[self.all_variables]
            y_test = test_set[self.target_variable]

            # Handle any missing values by filling with the mean of the column
            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)
            X_test.fillna(X_test.mean(), inplace=True)
            y_test.fillna(y_test.mean(), inplace=True)

        else:

            X, X_test, y, y_test = self.random_split()

        # Train the model
        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }

        # Initialize the model
        xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=50,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        # Fit the random search model
        random_search.fit(X, y)

        # Get the best parameters
        best_params = random_search.best_params_
        print("Best parameters found: ", best_params)

        # Train the model with the best parameters
        best_model = XGBRegressor(**best_params)
        best_model.fit(X, y)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring='neg_mean_absolute_error')

        print("Cross-Validation MAE Scores: ", -cv_scores)
        print("Mean CV MAE: ", -cv_scores.mean())
        print("Standard Deviation of CV MAE: ", cv_scores.std())

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.plot_with_intervals(X_test, y_test, y_pred)

        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared: {r2}")

        # Create box plot of cross-validated MAE scores
        cv_results = random_search.cv_results_
        mae_scores = -cv_results['mean_test_score']

        # Create a box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[mae_scores], color='#F4A460')
        plt.title('Mean Absolute Error')
        plt.ylabel('Mean Absolute Error')
        plt.show()

        # Extracting feature importance
        importances = best_model.feature_importances_
        features = X.columns

        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance from XGBoost Model')
        plt.gca().invert_yaxis()
        plt.show()

        # Save the model (optional)
        best_model.save_model('xgb_model.json')

        # plot the result
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_conditional_pred(y_test, y_pred, title="xgb", ax=ax)
        ax.set_ylabel(f'Predictions', fontsize=14)
        ax.set_xlabel('True values', fontsize=14)
        plt.savefig('xgb_prediction.png')

    def train_lgb(self, is_date_split=False):

        if is_date_split:

            X = self.df_train[self.all_variables]
            y = self.df_train[self.target_variable]

            X_test = self.df_test[self.all_variables]
            y_test = self.df_test[self.target_variable]

            # Handle any missing values by filling with the mean of the column
            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)
            X_test.fillna(X_test.mean(), inplace=True)
            y_test.fillna(y_test.mean(), inplace=True)

        else:

            X, X_test, y, y_test = self.random_split()

        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }

        # Initialize the model
        lgb = LGBMRegressor(objective='regression', random_state=42)

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=lgb,
            param_distributions=param_grid,
            n_iter=50,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        # Fit the random search model
        random_search.fit(X, y)

        # Get the best parameters
        best_params = random_search.best_params_
        print("Best parameters found: ", best_params)

        # Train the model with the best parameters
        best_model = LGBMRegressor(**best_params)
        best_model.fit(X, y)

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        print(y_pred)
        valid_idx = ~np.isnan(y_test) & ~np.isnan(y_pred)
        y_test_clean = y_test[valid_idx]
        y_pred_clean = y_pred[valid_idx]
        mse = mean_absolute_error(y_test_clean, y_pred_clean)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        # Extracting feature importance
        importances = best_model.feature_importances_
        features = X.columns

        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance from LightGBM Model')
        plt.gca().invert_yaxis()
        plt.show()

        # Save the model (optional)
        best_model.booster_.save_model('lgb_model.txt')

        # Plot the result
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_conditional_pred(y_test, y_pred, title="lgb", ax=ax)
        ax.set_ylabel(f'Predictions', fontsize=14)
        ax.set_xlabel('True values', fontsize=14)
        plt.savefig('lgb_prediction.png')

    def plot_with_intervals(self, test_x, true_y, pred_y):
        x = test_x['sars_cov2_gc_l_mean']
        true_y = true_y
        predict_y = pred_y

        print(x.shape, true_y.shape)

        # # Calculate confidence intervals
        # lower_90 = np.percentile(true_y, 5)
        # upper_90 = np.percentile(true_y, 95)
        # lower_50 = np.percentile(true_y, 25)
        # upper_50 = np.percentile(true_y, 75)

        plt.figure(figsize=(10, 10))

        # Scatter plot of true values
        plt.scatter(x, true_y, color='lightgreen', label='True values')

        # Scatter plot of predicted values
        plt.scatter(x, predict_y, color='red', label='Predicted values', marker='x')

        # Add labels and title
        plt.xlabel('RNA concentration (x)')
        plt.ylabel('Values')
        plt.legend()

        # Show the plot
        plt.show()

