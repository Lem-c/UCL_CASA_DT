import numpy as np
import geopandas as gpd
import pandas as pd
from util.logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns

cis_look_up_df_path = "./data/LAD19_CIS20_EN_LU.csv"


def date_parser(col_name):
    try:
        return pd.to_datetime(col_name).date()
    except ValueError:
        return col_name


def select_age(df, ageFrom):
    # Select the columns representing ages older than 'ageFrom'
    age_columns = [str(i) for i in range(ageFrom, 90)] + ['90+']
    col_name = 'over_' + str(ageFrom)
    # Sum the values of these columns for each row
    df[col_name] = df[age_columns].replace(',', '', regex=True).astype(int).sum(axis=1)

    return df


class DataBase:

    def __init__(self, df_vir: str, df_ww: str, sheet_var='Daily publication', sheet_ww='Regional_concentrations'):
        self.df_vir = None
        self.df_ww = None
        # The df used for regression
        self.df_main = None
        # The external df placeholder
        self.df_add = None
        # print tool
        self.log = Logger(__name__).get_logger()

        if len(df_vir) < 1 or sheet_var is None:
            self.df_main = pd.read_excel(df_ww)
        else:
            self.df_vir = pd.read_excel(df_vir, sheet_name=sheet_var, skiprows=12)
            self.df_ww = pd.read_excel(df_ww, sheet_name=sheet_ww, engine='odf', skiprows=7, header=0)

            self.setHeader()
            self.getDataInfo()

    def setHeader(self):
        """
        Clean the rows/columns not used for virus dataframe

        :return: None
        """
        self.df_vir.columns = ['Region'] + list(self.df_vir.columns[1:])
        # Drop the first column
        self.df_vir.drop(self.df_vir.columns[0], axis=1, inplace=True)
        # Rename the columns by stripping the time part
        self.df_vir.columns = [date_parser(col) for col in self.df_vir.columns]

        # Display the first few rows of the DataFrame
        self.log.info(self.df_vir.head())

    def getDataInfo(self):
        self.df_vir.info()
        self.df_ww.info()
        self.log.info(f"Shape of virus df: {self.df_vir.shape}")

    def pivot_df(self, target_area="england"):
        # Convert the data to the desired format and pivot the table
        self.df_ww.columns = pd.to_datetime(self.df_ww.columns[1:],
                                            format='%d/%m/%Y').strftime('%Y-%m-%d').insert(0, 'Region name')

        # Pivot the table
        pivoted_ww = self.df_ww.set_index('Region name').T.reset_index()
        pivoted_vir = self.df_vir.set_index('Name').T.reset_index()

        # Ensure all column names are strings
        pivoted_vir.columns = pivoted_vir.columns.map(str)
        # Get first cat data: 1. Total reported admissions to hospital and diagnoses in hospital
        pivoted_vir = pivoted_vir.iloc[:, 0:9]

        # drop NaN
        pivoted_ww = pivoted_ww.dropna()
        pivoted_vir = pivoted_vir.loc[:, ~pivoted_vir.columns.str.contains('nan')]

        self.log.info(pivoted_ww.head())
        self.log.info(pivoted_vir.columns)
        # Convert the column names to lower case
        pivoted_ww.columns = pivoted_ww.columns.str.lower()
        pivoted_vir.columns = pivoted_vir.columns.str.lower()

        pivoted_ww = pivoted_ww.melt(id_vars=['index'], var_name='region', value_name='ww_value')
        pivoted_vir = pivoted_vir.melt(id_vars=['index'], var_name='region', value_name='vir_value')

        # Filter to get only 'england' values
        pivoted_ww_target = pivoted_ww[pivoted_ww['region'] == target_area].drop(columns=['region'])
        pivoted_vir_target = pivoted_vir[pivoted_vir['region'] == target_area].drop(columns=['region'])

        self.log.info(pivoted_ww_target.head())
        self.log.info(pivoted_vir_target.head())

        # Convert 'index' columns to datetime
        pivoted_ww_target['index'] = pd.to_datetime(pivoted_ww_target['index'])
        pivoted_vir_target['index'] = pd.to_datetime(pivoted_vir_target['index'])

        # Convert 'ww_value' and 'vir_value' columns to numeric
        pivoted_ww_target['ww_value'] = pd.to_numeric(pivoted_ww_target['ww_value'], errors='coerce')
        pivoted_vir_target['vir_value'] = pd.to_numeric(pivoted_vir_target['vir_value'], errors='coerce')

        # merge england data
        self.df_main = pd.merge(pivoted_ww_target, pivoted_vir_target, on='index', how='inner')
        # self.df_main.to_csv("main.csv")

    def get_train_test_split(self) -> object:
        df_temp = self.df_main
        # Convert the date column to datetime format
        df_temp['date'] = pd.to_datetime(self.df_main['date'])

        # Sort the dataset by date
        data = df_temp.sort_values(by='date')

        # Calculate the 80% point
        split_point = int(len(data) * 0.8)

        # Split the dataset into training and testing sets
        train_set = data.iloc[:split_point]
        test_set = data.iloc[split_point:]

        return train_set, test_set

    def add_age_df(self, path, sort_column="Code"):
        """

        :param sort_column: The column in added df that used for merging
        :param path: csv recommend
        :return:
        """
        self.df_add = pd.read_csv(path)
        lookup_df = pd.read_csv(cis_look_up_df_path)

        # remove space
        self.df_add[sort_column] = self.df_add[sort_column].str.strip()

        if sort_column == 'Geography_code':
            # Find the denominator: Ethnic_Population for K04000001 with Ethnicity 'All'
            denominator = self.df_add[(self.df_add[sort_column] == 'K04000001') &
                                      (self.df_add['Ethnicity'] == 'All')]['Ethnic_Population'].values[0]
            # Calculate the proportion for each Ethnicity type
            self.df_add['Ethnicity_Proportion'] = self.df_add['Ethnic_Population'] / denominator

        # replace spatial unit to  CIS20CD
        merged_df = pd.merge(self.df_add, lookup_df, left_on=sort_column, right_on='LAD19CD')

        # Drop the unnecessary columns from the merge
        merged_df = merged_df.drop(columns=['LAD19NM', 'FID'])

        self.df_add = merged_df.reset_index(drop=True)

    def add_param2main_df(self, param: str, sort_column: str, name: str):
        """

        :param sort_column: The column in added df that used for merging
        :param param: Type of param will have been added
        :param name: The name of the column that has data of that param
        """
        if param == 'age':
            self.add_age_df("./data/London_2021_age.csv", sort_column)
            data = select_age(self.df_add, 50)
            data = data.dropna(axis=0).dropna(axis=1)
            data['All ages'] = data['All ages'].str.replace(',', '').astype(int)
            data['ratio_over_50'] = data['over_50'] / data['All ages']
            # merge
            self.df_main = self.df_main.merge(data[['CIS20CD', 'LAD19CD', name]], on='CIS20CD', how='left')
        if param == 'ethic':
            self.add_age_df("./data/population-by-ethnicity-and-local-authority-2021.csv", sort_column)
            data = self.df_add[self.df_add['Ethnicity'] == 'All']
            data = data.dropna(axis=0).dropna(axis=1)
            self.df_main = self.df_main.merge(data[['CIS20CD', name]], on='CIS20CD', how='left')
        if param == 'IMD':
            self.add_age_df('./data/ID_2019_London.csv', sort_column)
            data = self.df_add.groupby(
                'CIS20CD')[
                name].mean().reset_index()
            self.df_main = self.df_main.merge(data[['CIS20CD', name]], on='CIS20CD', how='left')

        self.df_main = self.df_main.dropna(axis=0).dropna(axis=1)
        self.log.info(self.df_main.head(10))
        self.df_add = None

    def calculate_confidence_intervals(self, n_resamples=1000):
        sars_cov2_gc_l_mean = self.df_main['sars_cov2_gc_l_mean'].values
        median_prob = self.df_main['infection_rate'].values

        def conf_interval(data, confidence=0.95):
            n = len(data)
            mean = np.mean(data)
            stderr = np.std(data, ddof=1) / np.sqrt(n)
            margin_of_error = stderr * 1.96  # 95% CI for a normal distribution
            return mean - margin_of_error, mean + margin_of_error

        ci_95_sars = conf_interval(sars_cov2_gc_l_mean, confidence=0.95)
        ci_50_sars = conf_interval(sars_cov2_gc_l_mean, confidence=0.50)
        ci_95_median = conf_interval(median_prob, confidence=0.95)
        ci_50_median = conf_interval(median_prob, confidence=0.50)

        return (ci_50_sars, ci_95_sars), (ci_50_median, ci_95_median)

    def plot_mean_vs_cov(self):
        # Group the data by 'CIS20CD' and aggregate the dates into a single row
        grouped_data = self.df_main.groupby('CIS20CD').agg({
            'date': lambda x: list(x),
            'sars_cov2_gc_l_mean': lambda x: list(x),
            'median_prob': lambda x: list(x)
        }).reset_index()

        # Function to add confidence intervals
        def add_confidence_intervals(ax, x, y, ci_base):
            # Add shaded areas for confidence intervals
            ax.fill_between(x, ci_base[0], ci_base[1], color='green', alpha=0.3,
                            label='95% CI')
            ax.fill_between(x, ci_base[2], ci_base[3], color='green', alpha=0.5,
                            label='50% CI')

        # Plot median_prob (y) vs. sars_cov2_gc_l_mean (x) for each group in separate subplots
        # with logarithmic scale and confidence intervals
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=True, sharey=True)

        # regions will be plotted
        subregions = ['J06000164', 'J06000165', 'J06000166']
        titles = ['Sub-region A', 'Sub-region B', 'Sub-region C']
        confidence_intervals = [1.2048, 49741.919, 1.2903, 56567.7827]

        for ax, subregion, title in zip(axes, subregions, titles):
            subset = self.df_main[self.df_main['CIS20CD'] == subregion]
            x = subset['sars_cov2_gc_l_mean']
            y = subset['median_prob']

            # Scatter plot
            ax.scatter(x, y, color='blue', label='observations')

            # Add confidence intervals
            x_sorted = np.sort(x)
            add_confidence_intervals(ax, x_sorted, y,
                                     [confidence_intervals[0] * x_sorted, confidence_intervals[1] * x_sorted,
                                      confidence_intervals[2] * x_sorted, confidence_intervals[3] * x_sorted])

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(title)
            ax.set_xlabel('RNA concentration (gc/L)')
            ax.set_ylabel('Prevalence (%)')

        # Add a single legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_london_map(self, param='infection_rate'):
        shapefile_path = './data/Covid_Infection_Survey_Dec_2020_UK_BUC/CIS_DEC_2020_UK_BUC.shp'
        gdf = gpd.read_file(shapefile_path)

        merged_gdf = gdf.merge(self.df_main, left_on='CIS20CD', right_on='CIS20CD')
        # Apply logarithmic transformation to 'median_prob'
        merged_gdf['log_median_prob'] = np.log1p(merged_gdf['infection_rate'])

        # Define a custom colormap from light blue to red
        colors = ["#ADD8E6", "#FFA500", "#FF0000"]  # Light blue to red
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'custom_cmap'
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        merged_gdf.plot(column=param, cmap=custom_cmap, ax=ax, legend=True,
                        legend_kwds={'label': "probability of being infected",
                                     'orientation': "horizontal"})
        ax.set_title('London CIS Region Filled with rate of being infected')
        ax.set_axis_off()
        plt.show()

    def plot_correction(self):
        # Select only numeric columns
        numeric_df = self.df_main.select_dtypes(include='number')
        numeric_df = numeric_df.drop(columns=['reac_vol_control'])

        corr_matrix = numeric_df.corr()

        # Plotting the correlation matrix
        plt.figure(figsize=(16, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_data_trend(self):
        # Convert the 'date' column to datetime format
        temp_df = self.df_main
        temp_df = temp_df.sort_values(by='date')
        # Calculate a 7-day moving average for smoother visualization
        temp_df['sars_cov2_gc_l_mean_ma'] = temp_df['sars_cov2_gc_l_mean'].rolling(window=7).mean()

        # Plot the original and smoothed trend of SARS-CoV-2 concentration over time
        plt.figure(figsize=(14, 7))
        plt.plot(temp_df['date'], temp_df['sars_cov2_gc_l_mean'], marker='o', linestyle='-', alpha=0.5,
                 label='Daily Mean Concentration')
        plt.plot(temp_df['date'], temp_df['sars_cov2_gc_l_mean_ma'], marker='', linestyle='-', color='r',
                 label='7-Day Moving Average')
        plt.title('Trend of SARS-CoV-2 Concentration Over Time')
        plt.xlabel('Date')
        plt.ylabel('SARS-CoV-2 Concentration (gc/l mean)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_box(self):
        num_columns = self.df_main.select_dtypes(include=['float64', 'int64']).columns
        num_plots = len(num_columns)

        # Set the number of rows to 2 and calculate the number of columns
        num_rows = 2
        num_cols = (num_plots // 2) + (1 if num_plots % 2 > 0 else 0)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        axes = axes.flatten()  # Flatten in case of single row

        for i, column in enumerate(num_columns):
            sns.boxplot(data=self.df_main, y=column, ax=axes[i])
            axes[i].set_title(f'{column}')

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_imd(self, london_isoa):
        # Load shapefile
        lsoa_gdf = gpd.read_file(london_isoa)

        # Load CSV data
        imd_data = pd.read_csv('./data/ID_2019_London.csv')

        # Merge the GeoDataFrame with the CSV data
        merged_gdf = lsoa_gdf.merge(imd_data, left_on="LSOA11CD", right_on="LSOA code (2011)")

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        merged_gdf.plot(column="imd_score", cmap="OrRd", linewidth=0.8, ax=ax, edgecolor="0.8", legend=True)

        # Add unique LA code annotations
        plotted_labels = set()
        for idx, row in merged_gdf.iterrows():
            la_code = row["Local Authority District name (2019)"]
            if la_code not in plotted_labels:
                plt.annotate(text=la_code, xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
                             horizontalalignment='center', fontsize=8, color='black')
                plotted_labels.add(la_code)

        plt.title("IMD Score in London")
        plt.show()

    def plot_result(self):
        dimensions = ['Original Data', 'ratio_over_50', 'Ethnicity_Proportion', 'imd_score']
        xgb_mae = [0.0549, 0.0499, 0.0521, 0.0528]
        xgb_r2 = [0.9230, 0.9326, 0.9286, 0.9282]
        lgb_mae = [0.0577, 0.0587, 0.0551, 0.0614]
        lgb_r2 = [0.9140, 0.9124, 0.9171, 0.9194]

        # Plotting MAE with different colors
        plt.figure(figsize=(10, 5))

        plt.plot(dimensions, xgb_mae, marker='o', label='XGB MAE', color='blue')
        plt.plot(dimensions, lgb_mae, marker='o', label='LGB MAE', color='green')

        plt.xlabel('Added socio-demographic variables')
        plt.ylabel('MAE')
        plt.title('MAE value of XGB and LGB output')
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)  # Increase the font size of x-axis tick labels
        plt.yticks(fontsize=12)  # Increase the font size of y-axis tick labels
        plt.grid(True)
        plt.tight_layout()

        # Show MAE plot
        plt.show()

        # Plotting R squared with different colors
        plt.figure(figsize=(10, 5))

        plt.plot(dimensions, xgb_r2, marker='o', label='XGB R squared', color='blue')
        plt.plot(dimensions, lgb_r2, marker='o', label='LGB R squared', color='green')

        plt.xlabel('Dimension')
        plt.ylabel('R squared')
        plt.title('R squared value of XGB and LGB prediction')
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)  # Increase the font size of x-axis tick labels
        plt.yticks(fontsize=12)  # Increase the font size of y-axis tick labels
        plt.grid(True)
        plt.tight_layout()

        # Show R squared plot
        plt.show()

    def get_main_df(self):
        return self.df_main
