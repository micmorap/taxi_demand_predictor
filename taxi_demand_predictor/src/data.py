from pathlib import Path
import requests
import pandas as pd

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Aim: Extract from requests package the taxis rides info into parquet format to save into data/raw folder.

    Args:
        year (integer): Required year to download info.
        month (integer): Required month to download info.
    
    Return:
        Path: Download and save parquet file according to year and month specified
    """
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = f'../data/raw/rides_{year}-{month:02d}.parquet'
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not available!!!')
    


def validate_raw_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Aim: Transform, validated and save rides dataframe in a month-year specified.

    Args:
        rides (Dataframe): Dataset required to rename columns and validate date ranges.
        year (integer): Required year to download info.
        month (integer): Required month to download info.
    
    Return:
        DataFrame saved, transformed and validated.

    """    
    rides = pd.read_parquet(f"../data/raw/rides_{year}-{month:02}.parquet")
    rides.head(5)
    rides = rides[['tpep_pickup_datetime', 'PULocationID']]
    
    rides.rename(columns={
        'tpep_pickup_datetime': 'pickup_datetime',
        'PULocationID': 'pickup_location_id',
    }, inplace=True)

    rides.head(5)
    
    year_month_start_limit = f"{year}-{month:02d}-01"
    year_month_final_limit = f"{year}-{month+1:02d}-01"

    rides = rides[rides.pickup_datetime >= year_month_start_limit]
    rides = rides[rides.pickup_datetime < year_month_final_limit]
    
    rides.pickup_datetime.describe()
    rides.to_parquet(f"../data/transformed/validated_rides_{year}-{month:02d}.parquet")

    return rides


