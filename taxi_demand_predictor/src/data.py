from pathlib import Path
from tqdm import tqdm
from typing import Optional, List

import requests
import pandas as pd
import plotly.express as px
import numpy as np

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

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
        path = RAW_DATA_DIR / f'/rides_{year}-{month:02d}.parquet'
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
    rides = pd.read_parquet(f"{RAW_DATA_DIR}/rides_{year}-{month:02}.parquet")
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

def get_cutoff_indices_features_and_target( data: pd.DataFrame, input_seq_len: int, step_size: int) -> list:
    """
    """
    stop_position = len(data) - 1
    
    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []
    
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices

def transform_ts_data_into_features_and_target(ts_data: pd.DataFrame, input_seq_len: int,
    step_size: int) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id, 
            ['pickup_hour', 'rides']
        ]

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(ts_data_one_location, input_seq_len, step_size)

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']
