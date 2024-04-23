from src.data import *

year = 2022
month = 2

data_raw_2022_2 = download_one_file_of_raw_data(year, month)


data_raw__validate2022_2 = validate_raw_data(data_raw_2022_2, year, month)
#data = pd.read_parquet("/Users/michaelandr/Desktop/Taxi_Prediction_Course_MLOps/taxi_demand_predictor/data/raw/rides_2022-02.parquet")
#print(data.head())