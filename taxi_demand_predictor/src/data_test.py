import pandas as pd


df = pd.read_parquet("/Users/michaelandr/Desktop/Taxi_Prediction_Course_MLOps/taxi_demand_predictor/data/transformed/ts_data_2022_01.parquet")

print(df.head())