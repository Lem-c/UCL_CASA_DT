from util import DataBase
from util import XGBModel

if __name__ == "__main__":
    db = DataBase.DataBase("./data/raw/COVID-19-daily-admissions-and-beds-20220512-211001-220331-v2.xlsx",
                           "./data/raw/Final_EMHP_wastewater_data_24022022-1.ods")

    db.pivot_df("england")
    df = db.get_main_df()

    model = XGBModel.XGB(df)
    model.train()
