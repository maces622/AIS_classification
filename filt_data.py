import os.path
import pickle

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from collections import namedtuple
import csv
import pickle as pkl

from scipy.interpolate import interp1d

vessel_dict={}
# [0 mmsi,1 TS,2 lat,3 lon,4 sog,5 cog,
# 6 length,7 width,8 draft,9vesselType]
source_data='california_1.pkl'
output_data='procd_california.pkl'
"""
该部分主要用于按照输入CNN的要求清理AIS数据流
分别按照速度、长度、目标船只类型进行清理
"""
ll_msg_2=[]
started_flg=False
now_mmsi=-1

def sog_jud(l_msg):
    max_sog=0.0
    sog_less_than_2=0
    for r in range(len(l_msg)):
        max_sog=max(max_sog,float(l_msg[r][4]))
        if float(l_msg[r][4])<=2.0:
            sog_less_than_2=sog_less_than_2+1
    
    if max_sog<=1.0 or float(sog_less_than_2)/float(len(l_msg))>0.7:
        return True
    else:
        return False
    
def len_jud(l_msg):
    delta_dt=l_msg[len(l_msg)-1][1]-l_msg[0][1]
    if delta_dt<timedelta(hours=6) or len(l_msg)<160:
        return True
    else :
        return False
def split_into_2(dt1,dt2):
    if dt2-dt1 > timedelta(hours=2) or (not (dt1.year == dt2.year and
                                        dt1.month == dt2.month and dt1.day == dt2.day)):
        # AIS数据点时间差大于2h or 不属于同一天
        # 需要分成两段轨迹

        return True
    else :
        return False

def in_lst(now_vesselType):
    target_lst=[30,1001,1002,
                60,61,62,63,64,65,66,67,68,69,1012,1013,1014,1015,
                70,71,72,73,74,75,76,77,78,79,1016,
                80,81,82,83,84,85,86,87,88,89,1017,1024]
    if now_vesselType in target_lst:
        return False
    else:
        return True


fishing = [30,1001,1002]
passgi=[60,61,62,63,64,65,66,67,68,69,1012,1013,1014,1015]
cargo =[70,71,72,73,74,75,76,77,78,79,1016]
tanker=[80,81,82,83,84,85,86,87,88,89,1017,1024]
def get_short_code(vessel_type):
    if vessel_type in fishing:
        return 3
    elif vessel_type in passgi:
        return 6
    elif vessel_type in cargo:
        return 7
    elif vessel_type in tanker:
        return 8



tot=0

with open(source_data,"rb") as f:
    pklReader=pkl.load(f)
    now_msg=[]
    for row in pklReader:
        row.append(tot)
        # print(type(row))
        if started_flg==False:
            now_msg=[]
            now_mmsi=row[0]
            now_msg.append(row)
            started_flg=True
            
        else:
            if row[0]==now_mmsi or not split_into_2(now_msg[-1][1],row[1]):
                now_msg.append(row)
            else :
                if len_jud(now_msg) or sog_jud(now_msg):
                    pass
                elif in_lst(now_msg[0][9]):
                    pass
                else:
                    now_msg[0][9]=get_short_code(now_msg[0][9])

                    ll_msg_2.append(now_msg)
                    tot = tot + 1
                    # print(now_msg[1])
                    if now_msg[0][9] in vessel_dict:
                        vessel_dict[now_msg[0][9]]+=1
                    else :
                        vessel_dict[now_msg[0][9]]=1
                    
                now_msg=[]
                now_mmsi=row[0]
                now_msg.append(row)

    # 记录最后一条AIS航迹
    if len_jud(now_msg) or sog_jud(now_msg):
        pass
    elif in_lst(now_msg[0][9]):
        pass
    else:
        now_msg[0][9] = get_short_code(now_msg[0][9])
        ll_msg_2.append(now_msg)
        tot = tot + 1
        if now_msg[0][9] in vessel_dict:
            vessel_dict[now_msg[0][9]] += 1
        else:
            vessel_dict[now_msg[0][9]] = 1

print(ll_msg_2[0])

                

print("合格的数据：",len(ll_msg_2))
print("总数据量：",tot)
# print(type(ll_msg_2))
sorted_vdict = sorted(vessel_dict.items(), key=lambda item: (item[0],item[1]))
print(sorted_vdict)

"""
IMO VESSEL TYPE LIST

fishing 30 1001 1002

====================================================================================

vessel type = 0 表示 不确定类型
[('73', 1), ('72', 1), ('54', 1), ('94', 1), ('81', 1), ('9', 1), 
 ('10', 1), ('35', 1), ('82', 1), ('84', 1), ('99', 1), ('79', 2), 
 ('36', 2), ('56', 2), ('20', 2), ('52', 3), ('71', 4), ('33', 4), 
 ('89', 5), ('37', 26), ('60', 88), ('30', 91), ('80', 119), ('31', 193), 
 ('0', 234), ('70', 256), ('90', 260)]
 

90 拖船、特殊船只 x

70 休闲船只 o
30、31、37 货船 o 
80 渔船 o
60 客船 o

0 未知 x
2x 油船 x

"""
# output :
# 合格的数据： 409
# 总数据量： 1301
# [(3, 112), (6, 33), (7, 175), (8, 89)]
"""
fixed_columns 是指不用插值的列c
interp_columns 是指的需要线性插值进行重新采样的列
# [0 mmsi,1 TS,2 lat,3 lon,4 sog,5 cog,
# 6 length,7 width,8 draft,9vesselType]

"""
def resample_data(data, target_length=160, fixed_columns=(0, 1, 6, 7, 8,9),
                  interp_columns=(2,3,4,5)):
    data_array=np.array(data)
    original_len=data_array.shape[0]
    num_cols=data_array.shape[1]
    new_len=target_length

    resampled_data = np.zeros((new_len, data_array.shape[1]),dtype=object)

    # 在指定的区间内返回均匀间隔的数字
    new_indices = np.linspace(0, original_len-1, num=new_len)

    for col in range(num_cols):
        if col in interp_columns:

            try:
                col_data = data_array[:, col].astype(float)
                f = interp1d(np.arange(original_len), col_data, kind='linear')
                resampled_data[:,col]=f(new_indices)
            except:
                print(f"Column {col} cannot be converted to float for interpolation. Skipping...")
        else:
            original_indices = np.arange(original_len)
            for i, new_idx in enumerate(new_indices):
                nearest_idx = np.argmin(np.abs(original_indices - new_idx))
                resampled_data[i, col] = data_array[nearest_idx, col]
    return resampled_data
#
ll_msg_3=[]
for track in ll_msg_2:

    resample_Track=(
        resample_data(track))
    ll_msg_3.append(resample_Track)
np_ll_msg=np.stack(ll_msg_3,axis=0)
# print(len(np_ll_msg[0]))
# print(type(np_ll_msg))
x=0
for track_num in range(np_ll_msg.shape[0]):
    if np_ll_msg[track_num,1,6]!=0:

        x=x+1
    for row in range(np_ll_msg.shape[1]):
        # print(np_ll_msg[track_num,row,1].timestamp())
        np_ll_msg[track_num,row,1]=np_ll_msg[track_num,row,1].timestamp()
        #
        # print(np_ll_msg[track_num,row,0])
        # print(np_ll_msg[track_num,row,6:8])
with open(output_data,'wb') as f:
    pkl.dump(np_ll_msg,f)
print("有静态参数的样本量：",x)
