from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm
from typing import Optional, List
import plotly.express as px


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


def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    
    location_ids = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(
        agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq='H')
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):

        # keep only rides for this 'location_id'
        agg_rides_i = agg_rides.loc[agg_rides.pickup_location_id == location_id, ['pickup_hour', 'rides']]
            
        # quick way to add missing dates with 0 in a Series
        # taken from https://stackoverflow.com/a/19324591
        agg_rides_i.set_index('pickup_hour', inplace=True)
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value=0)
        
        # add back `location_id` columns
        agg_rides_i['pickup_location_id'] = location_id

        output = pd.concat([output, agg_rides_i])
    
    # move the purchase_day from the index to a dataframe column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})
    
    return output
     

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Aim: Transform, validated and save rides dataframe in a month-year specified.

    Args:
        rides (Dataframe): Dataset required to set pickup_datetime hour rounded  and group by total rides per day.
    
    Return:
        DataFrame saved, transformed and validated.    
    """
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    #ts_data_path = f""
    #agg_rides_all_slots.to_parquet('../data/transformed/ts_data_2022_01.parquet')

    return agg_rides_all_slots
