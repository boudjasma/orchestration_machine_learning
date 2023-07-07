import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from kedro.extras.datasets.pandas import CSVDataSet
import gcsfs

#remove lines where passengers_count were not reprsented
def remove_passenger_count(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str,any]:
    def remove_pass_func(data: pd.DataFrame) ->pd.DataFrame:
        print(data['passenger_count'])
        index = data[(data['passenger_count'] == 7) | (data['passenger_count'] == 8) | (data['passenger_count'] == 9) | (data['passenger_count'] == 0)].index
        data.drop(index, inplace=True)
        return data
    return dict(train_remove_passenger=remove_pass_func(train), test_remove_passenger=remove_pass_func(test))


def remove_extrem_values(train: pd.DataFrame, ) -> pd.DataFrame:
    def remove_extrem_func(data: pd.DataFrame) ->pd.DataFrame:
        mean = np.mean(data['trip_duration'])
        standard_deviation = np.std(data['trip_duration'])
        data = data[data['trip_duration'].between(mean - 2 * standard_deviation, mean + 2 * standard_deviation)]
        return data
    return remove_extrem_func(train)


def map_store_and_fwd_flag(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str,any]:
    def map_func(data: pd.DataFrame) ->pd.DataFrame:
        data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)
        return data
    return dict(train_map_store=map_func(train), test_map_sotre=map_func(test))


def decompose_date(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str,any]:
    def decompose(data: pd.DataFrame) -> pd.DataFrame:
        data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
        if 'dropoff_datetime' in data.columns:
            data['dropoff_datetime'] = pd.to_datetime(data.dropoff_datetime)
            data['dropoff_date'] = data['dropoff_datetime'].dt.date
            data['dropoff_time'] = data['dropoff_datetime'].dt.time
            data['dropoff_hour'] = data['dropoff_datetime'].dt.hour
            data['dropoff_weekday'] = data['dropoff_datetime'].dt.weekday
            data['dropoff_month'] = data['dropoff_datetime'].dt.month
    
        data['pickup_date'] = data['pickup_datetime'].dt.date
        data['pickup_time'] = data['pickup_datetime'].dt.time
        data['pickup_weekday'] = data['pickup_datetime'].dt.weekday
        data['pickup_weekofyear'] = data['pickup_datetime'].dt.isocalendar().week
        data['pickup_hour'] = data['pickup_datetime'].dt.hour
        data['pickup_minute'] = data['pickup_datetime'].dt.minute
        data['pickup_dt'] = (data['pickup_datetime'] - data['pickup_datetime'].min()).dt.total_seconds()
        data['pickup_week_hour'] = data['pickup_weekday'] * 24 + data['pickup_hour']
        data['pickup_dayofyear'] = data['pickup_datetime'].dt.dayofyear
        data['pickup_month'] = data['pickup_datetime'].dt.month
        return data
    return dict(train_decompose_date=decompose(train), test_decompose_date=decompose(test))

def add_is_holidays(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str,any]:
    def add(data:pd.DataFrame) -> pd.DataFrame:
        calendar = USFederalHolidayCalendar()
        holidays = calendar.holidays()
        data['pickup_holiday'] = pd.to_datetime(data.pickup_datetime.dt.date).isin(holidays)
        data['pickup_holiday'] = data.pickup_holiday.map(lambda x: 1 if x == True else 0)
        data['pickup_near_holiday'] = (pd.to_datetime(data.pickup_datetime.dt.date).isin(holidays + timedelta(days=1)) | pd.to_datetime(data.pickup_datetime.dt.date).isin(holidays - timedelta(days=1)))
        data['pickup_near_holiday'] = data.pickup_near_holiday.map(lambda x: 1 if x == True else 0)        
        return data
    return dict(train_is_holiday=add(train), test_is_holiday=add(test))

def compute_distances(train: pd.DataFrame, test: pd.DataFrame) -> Dict[any,any]:
    def compute(data: pd.DataFrame) -> pd.DataFrame:
        # 1 Haversine Distance
        def haversine_distance(lat1, lng1, lat2, lng2):
            lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
            AVG_EARTH_RADIUS = 6371 
            lat = lat2 - lat1
            lng = lng2 - lng1
            d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
            h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
            return h

        # 2 Bearing Distance 
        def bearing_direction(lat1, lng1, lat2, lng2):
            AVG_EARTH_RADIUS = 6371
            lng_delta_rad = np.radians(lng2 - lng1)
            lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
            y = np.sin(lng_delta_rad) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
            return np.degrees(np.arctan2(y, x))

        # 3 Manhattan Distance
        def manhattan_distance(lat1, lng1, lat2, lng2):
            a = haversine_distance(lat1, lng1, lat1, lng2)
            b = haversine_distance(lat1, lng1, lat2, lng1)
            return a + b


        data['direction'] = bearing_direction(data['pickup_latitude'].values, data['pickup_longitude'].values, data['dropoff_latitude'].values, data['dropoff_longitude'].values)
        data['distance_haversine'] = haversine_distance(data['pickup_latitude'].values, data['pickup_longitude'].values, data['dropoff_latitude'].values, data['dropoff_longitude'].values)
        data['distance_manhattan'] = manhattan_distance(data['pickup_latitude'].values, data['pickup_longitude'].values, data['dropoff_latitude'].values, data['dropoff_longitude'].values)
        data['center_latitude'] = (data['pickup_latitude'].values + data['dropoff_latitude'].values) / 2
        data['center_longitude'] = (data['pickup_longitude'].values + data['dropoff_longitude'].values) / 2
    
        return data
    
    return dict(train_compute_distances=compute(train), test_compute_distances=compute(test))

#
def split_dataset(data: pd.DataFrame, test_ratio: float) -> Dict[str, any]:
    
    feature_names = list(data.columns)
    features_not_used = ['id', 'trip_duration_normalised', 'trip_duration', 'dropoff_datetime','dropoff_date','dropoff_hour',
                           'dropoff_month','dropoff_time','dropoff_weekday', 'pickup_date', 'pickup_datetime', 'date','pickup_time','pickup_month']
    feature_names = [f for f in data.columns if f not in features_not_used]
    
    X = data[feature_names]
    y = data["trip_duration"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=40
    )
    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
