import pandas as pd
from util.logger import Logger


def date_parser(col_name):
    try:
        return pd.to_datetime(col_name).date()
    except ValueError:
        return col_name


class DataBase:

    def __init__(self, df_vir: str, df_ww: str, sheet_var='Daily publication', sheet_ww='Regional_concentrations'):
        self.df_vir = None
        self.df_ww = None
        # The df used for regression
        self.df_main = None
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

    def get_main_df(self):
        return self.df_main
