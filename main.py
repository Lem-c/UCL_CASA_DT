from util import DataBase
from util import XGBModel

if __name__ == "__main__":
    db = DataBase.DataBase("",
                           './data/London_agg.xlsx')
    df_train, df_test = db.get_train_test_split()

    print(df_train)
    print("\n\n")
    print(df_test)

    xgb = XGBModel.XGB(df_train, df_test, db.get_main_df())
    xgb.train_lgb()
