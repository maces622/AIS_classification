import pandas as pd
from datetime import datetime
# MMSI	BaseDateTime LAT	LON	SOG	COG	Heading	VesselName	IMO	CallSign	VesselType	Status	Length	Width	Draft	Cargo	TransceiverClass
import csv
import pickle
import glob
"""
该部分代码主要对原始数据进行提取目标字段
按照MMSI和时间戳进行排序
"""
ll_msg_1=[]
lon_max=-90.0
lon_min=90.0
lat_max=-180
lat_min=180
sog_max=0.0
sog_min=1000.0
cog_max=0.0
cog_min=360.0

folder_path='./CA_DATA'
csv_file=glob.glob(f"{folder_path}/*.csv")
for file_n in csv_file:
    print(file_n)
    with open(file_n,'r') as f :
        csvReader=csv.reader(f)
        flg = 0
        for row in csvReader:
            if flg == 0:
                flg = 1
                continue
            mmsi = int(row[0])
            TS = datetime.strptime(str(row[1]), '%Y-%m-%dT%H:%M:%S')

            lat = float(row[2])
            lon = float(row[3])
            sog = float(row[4])
            cog = float(row[5])

            if row[10] != '':
                vesselType = int(row[10])
            else:
                vesselType = 0
            if row[12] != '':
                length = float(row[12])
            else:
                length = 0
            if row[13] != '':
                width = float(row[13])
            else:
                width = 0
            if row[14] != '':
                draft = float(row[14])
            else:
                draft = 0

            lat_max = max(lat_max, lat)
            lat_min = min(lat_min, lat)
            lon_max = max(lon_max, lon)
            lon_min = min(lon_min, lon)

            sog_max = max(sog_max, sog)
            sog_min = min(sog_min, sog)
            cog_max = max(cog_max, cog)
            cog_min = min(cog_min, cog)

            ll_msg_1.append([mmsi, TS, lat, lon, sog, cog, length, width, draft, vesselType])
    # break


print("开始排序")
# 自定义比较函数
def custom_sort(row):
    return (row[0], row[1])  # 按照第一个字段升序排列，若相同则按照第二个字段升序排列

# 对列表进行排序
sorted_data = sorted(ll_msg_1, key=custom_sort)
# print(sorted_data[:100])
with open("california_1.pkl",'wb') as f:
    pickle.dump(sorted_data,f)
# 使用 csv 模块写入数据到 CSV 文件


print("Data has been saved to", "california_1")





