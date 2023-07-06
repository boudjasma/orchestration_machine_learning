import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from pandas.tseries.holiday import USFederalHolidayCalendar

import seaborn as sns
import plotly.graph_objects as go
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans

# Read the train and test datasets
train_data = pd.read_csv('./data/split_train.csv', delimiter=";")
test_data = pd.read_csv('./data/split_test.csv', delimiter=";")

def remove_passenger_count(data: pd.DataFrame) -> pd.DataFrame:
    index = data[(data['passenger_count'] == 7) | (data['passenger_count'] == 8) | (data['passenger_count'] == 9) | (data['passenger_count'] == 0)].index
    data.drop(index, inplace=True)
    return data


def remove_extremvalues(data: pd.DataFrame) -> pd.DataFrame:
    mean = np.mean(data['trip_duration'])
    standard_deviation = np.std(data['trip_duration'])
    data = data[data['trip_duration'].between(mean - 2 * standard_deviation, mean + 2 * standard_deviation)]
    return data


def format_flags(data: pd.DataFrame) -> pd.DataFrame:
    data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)
    return data

def format_data(data: pd.DataFrame) -> pd.DataFrame:
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

def is_holidays(data:pd.DataFrame) -> pd.DataFrame:

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays()
    data['pickup_holiday'] = pd.to_datetime(data.pickup_datetime.dt.date).isin(holidays)
    data['pickup_holiday'] = data.pickup_holiday.map(lambda x: 1 if x == True else 0)
    data['pickup_near_holiday'] = (pd.to_datetime(data.pickup_datetime.dt.date).isin(holidays + timedelta(days=1)) | pd.to_datetime(data.pickup_datetime.dt.date).isin(holidays - timedelta(days=1)))
    data['pickup_near_holiday'] = data.pickup_near_holiday.map(lambda x: 1 if x == True else 0)
    
    return data

def compute_distances(data: pd.DataFrame) -> pd.DataFrame:
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

def split_dataset(test_data_set: pd.DataFrame) -> pd.DataFrame:




