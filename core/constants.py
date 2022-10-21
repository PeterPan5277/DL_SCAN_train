import os
from datetime import datetime, timedelta

def whole_day(year, month, day):
    ts = []
    dt = datetime(year, month, day)
    while dt.day == day:
        ts.append(dt)
        dt += timedelta(seconds=60 * TIME_GRANULARITY_MIN)
    return ts

def whole_hour(year, month, day, hour):
    ts = []
    dt = datetime(year, month, day, hour)
    while dt.hour == hour:
        ts.append(dt)
        dt += timedelta(minutes=10)
    return ts

def three_days(year, month, day):
    target_t = [datetime(year, month, day) + i * timedelta(days=1) for i in range(-1,2)]
    ts = []
    for calendar in target_t:
        ts.extend(whole_day(calendar.year, calendar.month, calendar.day))
    return ts
    

NX = 120
NY = 120
DBZ_Z = 21
TIME_GRANULARITY_MIN = 10
ROOT_DIR = os.environ['ROOT_DATA_DIR']
RADAR_DIR = f'{ROOT_DIR}/radar_2d_compressed/'
RAINFALL_DIR = f'{ROOT_DIR}/rain_compressed/'
#RAINFALL_RAW_DIR = f'{ROOT_DIR}/rain/test/'
# DATA_PKL_DIR = '/bk2/peterpan/PKL_2Dcrop_rain10m/'
DATA_PKL_DIR = '/bk2/peterpan/PKL_rain_radar/'
ERA_DIR = f'{ROOT_DIR}/era5/'
ERA_QV_DIR = f'{ROOT_DIR}/era5_qv/'


_dir_path = os.path.dirname(os.path.realpath(__file__))
TERRAIN_DIR = f'{os.path.dirname(_dir_path)}/Terrain.dat'

INPUT_LEN = 5
BALANCING_WEIGHTS = [1, 2, 5, 10, 30]

RAIN_Q95 = 10
RADAR_Q95 = 35
LOGALT_Q95 = 5.7

# Since those cases are what we interested, we pick out them from trainging dataset.
CASES_WE_WANT = three_days(2018,5,7) + three_days(2018,5,8) + three_days(2019,7,22) + three_days(2019,8,18) + \
                three_days(2019,9,30) + three_days(2019,10,1) + three_days(2019,12,30) + three_days(2019,12,31) + \
                three_days(2021,6,4) + three_days(2021,10,16)
CASES_WE_WANT = sorted(list(set(CASES_WE_WANT)))
                
# too large rainfall value (radar malfunction)
SKIP_TIME_LIST = CASES_WE_WANT + whole_day(2020,3,27) + whole_day(2020,3,28)
SKIP_TIME_LIST = sorted(list(set(SKIP_TIME_LIST)))
