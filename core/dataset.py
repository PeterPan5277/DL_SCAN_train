import calendar
import os,sys
import pickle
from datetime import datetime, timedelta
from multiprocessing import Pool
from tqdm import tqdm
import warnings
from collections import OrderedDict
import numpy as np
warnings.filterwarnings("ignore")

# os.environ['ROOT_DATA_DIR']='/bk2/handsomedong/DLRA_database'
# sys.path.extend(['/work/handsomedong/heterogeneous/DeepQPF_ChiaTung/training'])
from core.compressed_radar_data import CompressedRadarData
from core.compressed_rain_data import CompressedRainData
from core.constants import (DATA_PKL_DIR, RADAR_DIR, RAINFALL_DIR, CASES_WE_WANT,
                            SKIP_TIME_LIST, ERA_DIR, ERA_QV_DIR)
from core.file_utils import RadarFileManager, RainFileManager
from core.time_utils import TimeSteps
from core.specific_humidity import lookForNearestTime

from core.findSkipTime import findSkipTime_all, tooLargeRadar


def create_dataset(
        start_dt,
        end_dt,
        input_len,
        target_len,
        radar_dir=RADAR_DIR,
        rain_dir=RAINFALL_DIR,
        disjoint_entries=False,
):
    radar_fm = RadarFileManager(radar_dir)
    rain_fm = RainFileManager(rain_dir)
    cur_dt = start_dt
    dt_list = [cur_dt]
    while end_dt > cur_dt:
        cur_dt = TimeSteps.next(cur_dt)
        if cur_dt in SKIP_TIME_LIST:
            continue
        dt_list.append(cur_dt)

    dataset = []
    N = len(dt_list) - (input_len - 1) - target_len
    stepsize = 1
    if disjoint_entries:
        stepsize = input_len

    for i in range(0, N, stepsize):
        inp_radar_fpaths = [radar_fm.fpath_from_dt(dt) for dt in dt_list[i:i + input_len]]
        inp_rain_fpaths = [rain_fm.fpath_from_dt(dt) for dt in dt_list[i:i + input_len]]
        inp = list(zip(inp_radar_fpaths, inp_rain_fpaths))

        target = [rain_fm.fpath_from_dt(dt) for dt in dt_list[i + input_len:i + input_len + target_len]]
        dataset.append((inp, target))

    print(f"Created Dataset:{start_dt.strftime('%Y%m%d_%H%M')}-{end_dt.strftime('%Y%m%d_%H%M')}, "
          f" Disjoint:{int(disjoint_entries)} InpLen:{input_len} TarLen:{target_len} {len(dataset)}K points")

    return dataset


def keep_first_half(dic):
    """
    Keep data only for first 15 days of each month.
    """
    output = {k: v for k, v in dic.items() if k.day <= 15}
    return output


def keep_later_half(dic):
    """
    Keep only that data which is not being kept in keep_first_half()
    """
    fh_dict = keep_first_half(dic)
    return {k: v for k, v in dic.items() if k not in fh_dict}

def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, 
    # or programattically said, the previous day of the first of next month
    return next_month - timedelta(days=next_month.day) + timedelta(hours=23, minutes=50)

def load_from_list(dradar_test, drain_test, dscan_test, dt_pkl:list, resume_list:list) -> (dict):
    year = 0; month = 0
    for dt in resume_list:
        if dt not in dt_pkl:
            if (year != dt.year) or (month != dt.month):
                year = dt.year
                month = dt.month
                start_dt = datetime(year, month, 1)
                end_dt = last_day_of_month(start_dt)
                fname = os.path.join(DATA_PKL_DIR, 'AllDataDict_{start}_{end}.pkl')
                fname = fname.format(start=start_dt.strftime('%Y%m%d-%H%M'),
                                     end=end_dt.strftime('%Y%m%d-%H%M'),)
                print(fname)
                with open(fname, 'rb') as f:
                    output = pickle.load(f)
            try:

                dradar_test[dt] = output["radar"][dt]
                drain_test[dt] = output["rain"][dt]
                dscan_test[dt] = output["scan"][dt]
            except:
                continue
    #return(dradar_test, drain_test, dscan_test)
            #這邊卻尚未return，之後再補
      
def load_data(start_dt, end_dt, is_validation=False, is_test=False, is_train=False, workers=0, missing_dt=[]):
    assert int(is_validation) + int(is_test) + int(is_train) == 1, 'Data must be either train, test or validation'
    dtype_str = ['Train'] * int(is_train) + ['Validation'] * int(is_validation) + ['Test'] * int(is_test)
    print(f'[Loading {dtype_str[0]} Data] {start_dt} {end_dt}')

    arguements = []
    assert start_dt < end_dt
    cur_dt = start_dt
    while cur_dt < end_dt:
        # Ashesh想要一個月一個月的存資料
        last_day_month = calendar.monthrange(cur_dt.year, cur_dt.month)[1]
        # NOTE: 23:50 is the last event. this may change if we change the granularity
        offset_min = 23 * 60 + 50 - (cur_dt.hour * 60 + cur_dt.minute)
        cur_end_dt = min(end_dt, cur_dt + timedelta(days=last_day_month - cur_dt.day, seconds=60 * offset_min))
        arguements.append((cur_dt, cur_end_dt))
        cur_dt = TimeSteps.next(cur_end_dt) #跳下一筆資料，i.e.10分鐘
    #print('Months I want are:', arguements)
    #arguements存的類似[(2022/1/1, 2022/1/31), (2022/2/1, 2022/2/28), .......]

    
    data_dicts = []
    if workers > 0:
        with Pool(processes=workers) as pool:
            with tqdm(total=len(arguements)) as pbar:
                for i, data_dict in enumerate(pool.imap_unordered(_load_data, arguements)):
                    if data_dict.get('fpath'):
                        pbar.set_description(f"Loaded from {data_dict['fpath']}")
                    pbar.update()
                    data_dicts.append(data_dict)
    else:
        for args in tqdm(arguements):
            data_dicts.append(_load_data(args, missing_dt))
            
    
    radar_dict = {}
    rain_dict = {}
    scan_dict = {}#
    for d in data_dicts:
        d = comb_n_kick(d, SKIP_TIME_LIST, CASES_WE_WANT, test=is_test, dev=is_validation, train=is_train)
        radar_dict = {**radar_dict, **d['radar']}
        rain_dict = {**rain_dict, **d['rain']}
        scan_dict = {**scan_dict,**d['scan']}#
    
    if is_test:
        dt_pkl = list(radar_dict.keys())
        load_from_list(radar_dict, rain_dict, scan_dict, dt_pkl, CASES_WE_WANT)
        #radar_dict, rain_dict, scan_dict = load_from_list(radar_dict, radar_dict, scan_dict, dt_pkl, CASES_WE_WANT)

    return {'rain': rain_dict, 'radar': radar_dict, 'scan': scan_dict}#
def _load_data(args, Missing=[]):
    start_dt, end_dt = args
    #print(start_dt) #2015-01-01 0000Z(datetime)
    #print(start_dt.strftime('%Y/%m/%Y%m%d/%H%M')) #2015/01/20150101/0000
    fname = os.path.join(DATA_PKL_DIR, 'AllDataDict_{start}_{end}.pkl')
    fname = fname.format(
        start=start_dt.strftime('%Y%m%d-%H%M'),
        end=end_dt.strftime('%Y%m%d-%H%M'),
    )
   #此為原先版本
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            output = pickle.load(f)
        return output
    #(下段是為了多把scan data 掛上去)
    #if exitst(因為原先已經做好radar and rain)，(再把scan datadump上去)
    ##先把dt_list做好
    # cur_dt = start_dt
    # dt_list = []
    # while end_dt >= cur_dt:
    #     if cur_dt not in Missing:
    #         dt_list.append(cur_dt)
    #         cur_dt = TimeSteps.next(cur_dt)

    # scan_data = {}
    # for dt in dt_list:
    #    scan_data[dt] = _load_scan_data(dt)

    # scan_dic = {'scan': scan_data}


    # if os.path.exists(fname):
    #     with open(fname, 'rb') as f:
    #         original_dic = pickle.load(f)
    #         original_dic.update(scan_dic)  #新做好的scan dict
    #     with open(fname, 'wb') as f:
    #         pickle.dump(original_dic,f)
    #         output = original_dic

    #     return output
    #以下是for 如果ALLDATADICT連radar/rain都沒有時，就是要完全重做
    # radar_fm = RadarFileManager(RADAR_DIR, compressed=True)
    # rain_fm = RainFileManager(RAINFALL_DIR, compressed=True)

    # radar_data = {}
    # rain_data = {}
    # for dt in dt_list:    
    #     radar_data[dt] = CompressedRadarData(radar_fm.fpath_from_dt(dt)).load_raw() # [x, y, value][3, n]
    #     rain_data[dt] = CompressedRainData(rain_fm.fpath_from_dt(dt)).load_raw(can_raise_error=True)

    # output = {'radar': radar_data, 'rain': rain_data}
    # with open(fname, 'wb') as f:
    #     pickle.dump(output, f)

    # return output

def _load_scan_data(dt):
    # 2022/08/30 我這邊gygx只是先改好 事實上還沒重新做好alldatadict
    #這邊要是不存在就補120*120的0矩陣進去，若存在就回傳
    path_end = dt.strftime('%Y/%m/%Y%m%d/%H%M')  #2015/01/20150101/0000
    path_pkl = '/bk2/peterpan/SCAN_processed/TAHOPE_ONLYgrid_renew/'+path_end
    hhmm = path_pkl[-4:]
    path_dir = os.path.join(path_pkl,f'{hhmm}.pkl')
    
    if os.path.exists(path_dir):
        with open(path_dir, 'rb') as f:
            data_dic = pickle.load(f)
            gx=data_dic['grid_x']
            gy=data_dic['grid_y']
            #此處需要儲存順序為y, x，和模式相同
            gygx = [gy,gx]      
        return(gygx)
def comb_n_kick(data, remove_all:list, leave_alone:list, test=False, dev=False, train=False):
    #kick的部分是有些我們有興趣的個案想要拿來test但是出現在2015-2018訓練資料中，
    #或者存在在validation資料中，我們在train and valida先剔除他們後
    #再從test把他加上去這些資料
    if train:
        dt_pkl = list(data['radar'].keys())
        for elimi in remove_all:
            if elimi in dt_pkl:
                del data['radar'][elimi]
                del data['rain'][elimi]
                del data['scan'][elimi] #new added scan
    elif dev: # first-half month for valid
        data = {'radar': keep_first_half(data['radar']), 
                'rain': keep_first_half(data['rain']),
                'scan': keep_first_half(data['scan'])} #new added scan
        dt_pkl = list(data['radar'].keys())
        for elimi in remove_all:
            if elimi in dt_pkl:
                del data['radar'][elimi]
                del data['rain'][elimi]
                del data['scan'][elimi] #new added scan
    elif test:
        data = {'radar': keep_later_half(data['radar']),
                'rain': keep_later_half(data['rain']),
                'scan': keep_later_half(data['scan'])}
        # load the missing cases we want
        dt_pkl = list(data['radar'].keys())
        for elimi in remove_all:
            if (elimi in dt_pkl) & (elimi not in leave_alone): #只刪除資料犦掉的那些
                del data['radar'][elimi]
                del data['rain'][elimi]
                del data['scan'][elimi] #new added scan
    return data

if __name__ == '__main__':
    #%% skip time for 2019.01-2020.10
    # missing data
    # must ends with year
    filepath_radar = '/bk2/handsomedong/DLRA_database/radar_2d_compressed/2018'
    filepath_rain = '/bk2/handsomedong/DLRA_database/rain_compressed/2018'

    missing = findSkipTime_all(filepath_radar, datetime(2018,1,1), datetime(2018,12,31,23,50))
    missing += findSkipTime_all(filepath_rain, datetime(2018,1,1), datetime(2018,12,31,23,50))

    # too large number @ radar
    missing += tooLargeRadar(filepath_radar, workers=8)

    missing = sorted(list(set(missing)))
    ###
    start_dt = datetime(2018, 1, 1)
    end_dt = datetime(2018, 12, 31, 23, 50)
    load_data(start_dt, end_dt, is_test=True, workers=0, missing_dt=missing)
