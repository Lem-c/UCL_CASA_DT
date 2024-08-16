from util import DataBase
from util import XGBModel

if __name__ == "__main__":
    db = DataBase.DataBase("",
                           './data/London_agg.xlsx')
    df_train, df_test = db.get_train_test_split()
    print(db.calculate_confidence_intervals())
    db.add_param2main_df('age', 'Code', 'ratio_over_50')
    db.add_param2main_df('ethic', 'Geography_code', 'Ethnicity_Proportion')
    db.add_param2main_df('IMD', 'la_code', 'imd_score')

    # db.plot_box()
    # db.plot_data_trend()
    # db.plot_correction()
    # db.plot_london_map()
    # db.plot_imd('./data/london_LSOA/LSOA_2011_London_gen_MHW.shp')

    xgb = XGBModel.XGB(df_train, df_test, db.get_main_df())
    xgb.train_xgb(is_date_split=False, is_spatial_split=True)  # record MAE: [0.0528]; R2: [0.9282] =>
    # xgb_0_d [0.0549, 0.9230]
    # xgb_1_d [0.0499, 0.9326]
    # xgb_2_d [0.0521, 0.9286]

    # spatial [0.0811, 0.9298] nolog [ 0.2376, 0.8503]



    print("--")
    # xgb.train_lgb(is_date_split=False)  # record MAE: [0.0614]; R2: [0.9194]

    # lgb_0_d [0.0577, 0.9140]
    # lgb_1_d [0.0587, 0.9124]
    # lgb_2_d [0.0551, 0.9171]

    db.plot_result()

