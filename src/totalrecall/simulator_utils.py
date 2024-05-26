########################################################################
# Author(s):    Ashwin Kanhere
# Date:         21 September 2021
# Desc:         Code to save simulated measurements from zigzag or 
#               previously simulate trajectories
########################################################################

import numpy as np
import pandas as pd
import os
import pytz
from datetime import datetime

from gnss_lib.ephemeris_manager import EphemerisManager 
import gnss_lib.sim_gnss as sim_gnss
from gnss_lib.utils import datetime_to_tow
from gnss_lib.coordinates import geodetic2ecef
from totalrecall.traject_utils import traject_gen_zigzag, traject_load_matlab
# 生成模拟GNSS数据：函数名称、轨迹步数、起始时间、起始位置ECEF坐标、存储星历数据目录、保存数据块大小、轨迹索引
# save是否保存、savepath保存路径、loadpath加载轨迹数据路径、noise是否添加噪声
def save_simulated_dataset(traject_func, traject_steps, start_time, start_ECEF, ephemeris_data_directory, chunk_size=100, traject_idx=0, save=True, savepath=None, loadpath=None, noise=True):
    # 检查和加载轨迹数据
    if traject_func != "traject_gen_zigzag" and traject_func != "MATLAB_saved":
        raise NotImplementedError
    elif traject_func == "MATLAB_saved":
        if loadpath:
            time_vec, ECEF_traject = traject_load_matlab(loadpath, traject_idx, start_time, start_ECEF)
        else:
            ValueError('Provide directory to load trajectories from')
    elif traject_func == "traject_gen_zigzag":
        time_vec, ECEF_traject = traject_gen_zigzag(start_time, start_ECEF, traject_steps)
    save_data = []
    if traject_steps is None:
        len_traject = len(ECEF_traject)
    else:
        len_traject = np.min((len(ECEF_traject), traject_steps))
    sats = ['G'+"%02d"%sv_num for sv_num in range(1, 33)]
    manager = EphemerisManager(ephemeris_data_directory)

    if savepath is None:
        print('Enter valid filepath for saving')
    if savepath.endswith('.csv'):
        print('Enter a valid rootpath for saving and not a file name')
    # 生成和保存模拟数据
    for idx in range(len_traject):
        # 每100步打印一次进度
        if np.mod(idx, 100)==0:
            print('Running step number ', idx)
        # There was a datatype issue caused by pd.datetime objects in the following line so used a naive datetime instead
        ephemeris = manager.get_ephemeris(time_vec[idx], sats)
        gpsweek, tow = datetime_to_tow(time_vec[idx])  # 当前时间转换为GPS周和周内秒
        # 轨迹数据中获取当前位置和当前速度
        pos = np.array([ECEF_traject.loc[idx, 'Rxx'], ECEF_traject.loc[idx, 'Rxy'], ECEF_traject.loc[idx, 'Rxz']],dtype=float)
        vel = np.array([ECEF_traject.loc[idx, 'Rxvx'], ECEF_traject.loc[idx, 'Rxvy'], ECEF_traject.loc[idx, 'Rxvz']], dtype=float)
        # 噪声设置
        if noise:
            prange_sigma = 6
            doppler_sigma = 0.1
        else:
            prange_sigma = 0
            doppler_sigma = 0
        measurements, satXYZV = sim_gnss.simulate_measures(gpsweek, tow, ephemeris, pos, ECEF_traject.loc[idx, 'b'], ECEF_traject.loc[idx, 'b_dot'], vel, prange_sigma=prange_sigma, doppler_sigma=doppler_sigma)
        keep_ephem = ephemeris.loc[measurements.index]  # 保留与测量相关星历数据
        
        traject_series = ECEF_traject.iloc[idx]  # 获取当前时间步长的轨迹数据
        time_df = pd.DataFrame(np.tile(time_vec[idx], (len(measurements.index), 1)), columns=['t_idx'])
        traject_df = pd.DataFrame(np.tile(traject_series.values, (len(measurements.index), 1)), columns=traject_series.index)
        # 合并：时间、轨迹、测量数据、卫星数据和星历数据
        save_frame = pd.concat([time_df, traject_df, measurements.reset_index(), satXYZV.reset_index(drop=True), keep_ephem.reset_index(drop=True)], axis=1)
        save_data.append(save_frame)
        if np.mod(len(save_data), chunk_size) == 0:
            save_data = pd.concat(save_data)
            save_data = save_data.reset_index(drop=True)
            if save:
                chunk_num_str = "{:03d}".format(int(np.around(idx/chunk_size)))  # 生成编号字符串
                savename = savepath + str(traject_idx) + "_" + chunk_num_str + "_1.csv"  # 生成保存文件路径
                save_data.to_csv(savename)  # 保存至CSV文件中
                print('Saved chunk number ' + chunk_num_str)
            save_data = []
    # Save tail of the dataset保存剩余数据
    if save_data:
        save_data = pd.concat(save_data)
        save_data = save_data.reset_index(drop=True)
        if save:
            chunk_num_str = "{:03d}".format(int(np.around(idx/chunk_size)))
            savename = savepath + str(traject_idx) + "_" + chunk_num_str + "_1.csv"
            save_data.to_csv(savename)
            print('Saved chunk number ' + chunk_num_str)
    return save_data
