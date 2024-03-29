import pandas as pd
from datetime import datetime,timedelta
from collections import namedtuple
import csv
import pickle as pkl

vessel_dict={}
# [0 mmsi,1 TS,2 lat,3 lon,4 sog,5 cog,6 vesselType,
# 7 length,8 width,9 draft]
source_data='output_p.pkl'
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

def in_lst(now_vesselType):
    target_lst=[7,3,8,6]
    high_bit=int(now_vesselType/10)
    if high_bit in target_lst:
        return False
    else:
        return True

def cl(now_vesselType):
    mainType=int(now_vesselType/10)
    return mainType


tot=0

with open(source_data,"rb") as f:
    pklReader=pkl.load(f)
    now_msg=[]
    for row in pklReader:
        if started_flg==False:
            now_msg=[]
            now_mmsi=row[0]
            now_msg.append(row)
            started_flg=True
            
        else:
            if row[0]==now_mmsi:
                now_msg.append(row)
            else :
                tot=tot+1
                if len_jud(now_msg) or sog_jud(now_msg):
                    pass
                elif in_lst(now_msg[0][6]):
                    pass
                else:
                    now_msg[0][6]=cl(now_msg[0][6])
                    ll_msg_2.append(now_msg)

                    if now_msg[0][6] in vessel_dict:
                        vessel_dict[now_msg[0][6]]+=1
                    else :
                        vessel_dict[now_msg[0][6]]=1
                    
                now_msg=[]
                now_mmsi=row[0]
                now_msg.append(row)
                

print("合格的数据：",len(ll_msg_2))
print("总数据量：",tot)

sorted_vdict = sorted(vessel_dict.items(), key=lambda item: (item[0],item[1]))
print(sorted_vdict)

"""
    20-29: 油轮
    30-39: 货船
    40-49: 高速船（包括快艇、高速客货船等）
    50-59: 渡轮
    60-69: 客船（包括游轮和其他大型客船）
    70-79: 休闲船只（包括帆船、游艇等）
    80-89: 渔船
    90-99: 拖船和特殊船只（例如搜救船、拖轮等）

不合格的数据： 1262
总数据量： 1301

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
"""
[(73, 1), (72, 1), 
(81, 1), (33, 1), (82, 1), (84, 1), 
(79, 2), (36, 2), (71, 4), (37, 4), 
(89, 5), (60, 33), (30, 39), (31, 66), 
(80, 81), (70, 167)]

"""

# 合格的数据： 409
# 总数据量： 1301
# [(3, 112), (6, 33), (7, 175), (8, 89)]
