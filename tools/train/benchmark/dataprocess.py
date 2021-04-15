# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import pandas as pd
import sys
import numpy as np
np.__version__

# %%
# set parameters
para_num = len(sys.argv) # para_num check
if para_num != 3:
    print("usage: python dataprocess.py DeviceName NetName")
    sys.exit(0)

DeviceName = sys.argv[1]
NetName = sys.argv[2]

print(DeviceName)
print(NetName)

BatchSize = [1, 2, 4, 8, 16]
print(BatchSize)

path_to_usage_monitor = "./" + DeviceName + "/usage_monitor/usage_monitor_" + NetName + ".result"
path_to_expr_bench = "./" + DeviceName + "/train_bench/train_bench_" + NetName + ".result"
path_to_session_bench = "./" + DeviceName + "/bench_session.result"
path_to_freq = "./freq.result"

path_to_processed_data = "./" + DeviceName + "/processed_data/" + NetName + "/processed_data_" + NetName + ".csv"
path_to_time_table = "./" + DeviceName + "/processed_data/" + NetName + "/time_table_" + NetName + ".csv"
path_to_performance = "./" + DeviceName + "/processed_data/" + NetName + "/performance_" + NetName + ".csv"


# %%
# load raw data and extract their attributes into csv file
raw_data = pd.read_table(path_to_usage_monitor, header=None, sep=",", engine='python')

attributes= ["stamp", "processPid", "VIRT", "RES", "SHR", "cpuUsage", "battery_current", "battery_voltage", "usb_current", "usb_voltage" ]
processed_data = pd.DataFrame(index=attributes)

data = pd.Series(["nan"]*10, attributes)

count = 0
for line in raw_data[0]:
    if "NEW DATA" in line:
        count = count + 1
        data["stamp"] = line.split()[2].split("N")[0]
    if "battery_current" in line:
        data["battery_current"] = line.split()[1]
    if "battery_voltage" in line:
        data["battery_voltage"] = line.split()[1]
    if "usb_current" in line:
        data["usb_current"] = line.split()[1]
    if "usb_voltage" in line:
        data["usb_voltage"] = line.split()[1]
    if "status of pid" in line:
        data["processPid"] = line.split()[3]
    if "shell" in line:
        temp = line.split()
        shell_loc = temp.index("shell")
        data["VIRT"] = line.split()[3+shell_loc]
        if "G" in data["VIRT"]: # 将G转化为M
            data["VIRT"] = str(float(data["VIRT"].split("G")[0])*1024) + "M"
        data["RES"] = line.split()[4+shell_loc]
        if "G" in data["RES"]: # 将G转化为M
            data["RES"] = str(float(data["RES"].split("G")[0])*1024) + "M"
        data["SHR"] = line.split()[5+shell_loc]
        if "G" in data["SHR"]: # 将G转化为M
            data["SHR"] = str(float(data["SHR"].split("G")[0])*1024) + "M"
        data["cpuUsage"] = line.split()[7+shell_loc]
    if "END" in line:
        processed_data[count] = data
        data[:] = "nan"

matrix = processed_data.T

matrix.to_csv(path_to_processed_data)

    


# %%
# running time of expr (train + infer) and session (infer)
expr_data = pd.read_table(path_to_expr_bench, header=None, delimiter="\t")
session_data = pd.read_table(path_to_session_bench, header=None, delimiter="\t", engine='python')

attributes= ["batchsize", "expr_train", "expr_infer", "session_infer"]
time = pd.DataFrame(index=attributes)
data = pd.Series(["nan"]*4, attributes)

count = 0
for batchsize in BatchSize:
    count = count + 1
    data["batchsize"] = batchsize
    keywords_expr = "(batchsize is " + str(batchsize) + ")"
    keywords_session = "." + NetName + "_" + str(batchsize) + ".mnn"
    print(keywords_session)
    for line in expr_data[0]:
        #print(type(line))
        if keywords_expr in line:
            if "Training" in line:
                data["expr_train"] = line.split()[6]
            if "Inferring" in line:
                data["expr_infer"] = line.split()[6]
    for line in session_data[0]:
        if keywords_session in line:
            data["session_infer"] = line.split()[12].split("ms")[0]
    time[count] = data
    data[:] = "nan"

matrix = time.T
matrix.to_csv(path_to_time_table)


# %%
# read data from processed_data and analyse it with time_stamp
processed_data = pd.read_table(path_to_processed_data, header=0, index_col=0, sep=",")
attributes= ["batchsize", "VIRT", "RES", "SHR", "cpuUsage_train", "cpuUsage_infer", "battery_current", "battery_voltage", "usb_current", "usb_voltage" ]
performance = pd.DataFrame(index=attributes)
perform_temp = pd.Series([0]*10, attributes) # 初始化性能值存储条

# 记录训练、推断过程的时间戳
trainStart = 0
trainEnd = 0
inferStart = 0
inferEnd = 0

num = 0
for batchsize in BatchSize:
    perform_temp["batchsize"] = batchsize
    num = num + 1
    path_to_stamp = "./" + DeviceName + "/train_stamp/" + NetName + "/" + "train_stamp_" + str(batchsize) + ".result" 
    data = pd.read_table(path_to_stamp, header=None, delimiter="\t")
    for line in data[0]:
        if "Begin training" in line:
            trainStart = int(line.split()[3].split("N")[0])
        if "End trainning" in line:
            trainEnd = int(line.split()[3].split("N")[0])
        if "Begin inferring" in line:
            inferStart = int(line.split()[3].split("N")[0])
        if "End inferring" in line:
            inferEnd = int(line.split()[3].split("N")[0])

    count = 0 # 用来记录一个阶段里出现的记录条数
    for indexs in processed_data.index: # 按行对数据进行遍历
        if int(processed_data.loc[indexs]["stamp"]) > trainStart and int(processed_data.loc[indexs]["stamp"]) < trainEnd:
            if str(processed_data.loc[indexs]["cpuUsage"]) == "nan" or str(processed_data.loc[indexs]["cpuUsage"]) == "R": 
                continue
            count = count + 1 
            perform_temp["cpuUsage_train"] = perform_temp["cpuUsage_train"] + processed_data.loc[indexs]["cpuUsage"]
            # 内存数据有M后缀，需处理一下
            perform_temp["VIRT"] = max (float(processed_data.loc[indexs]["VIRT"].split("M")[0]), perform_temp["VIRT"])
            perform_temp["RES"] = max (float(processed_data.loc[indexs]["RES"].split("M")[0]), perform_temp["RES"])
            #perform_temp["SHR"] = max (float(processed_data.loc[indexs]["SHR"].split("M")[0]), perform_temp["SHR"])
            perform_temp["battery_current"] = max (int(processed_data.loc[indexs]["battery_current"]), perform_temp["battery_current"])
            perform_temp["battery_voltage"] = max (int(processed_data.loc[indexs]["battery_voltage"]), perform_temp["battery_voltage"])
            perform_temp["usb_current"] = max (int(processed_data.loc[indexs]["usb_current"]), perform_temp["usb_current"])
            perform_temp["usb_voltage"] = max (int(processed_data.loc[indexs]["usb_voltage"]), perform_temp["usb_voltage"])
    perform_temp["cpuUsage_train"] = perform_temp["cpuUsage_train"]/count

    count = 0 
    for indexs in processed_data.index: # 按行对数据进行遍历
        if int(processed_data.loc[indexs]["stamp"]) > inferStart and int(processed_data.loc[indexs]["stamp"]) < inferEnd:
            if str(processed_data.loc[indexs]["VIRT"]) == "nan": 
                continue
            count = count + 1 
            perform_temp["cpuUsage_infer"] = perform_temp["cpuUsage_infer"] + processed_data.loc[indexs]["cpuUsage"]
            perform_temp["VIRT"] = max (float(processed_data.loc[indexs]["VIRT"].split("M")[0]), perform_temp["VIRT"])
            perform_temp["RES"] = max (float(processed_data.loc[indexs]["RES"].split("M")[0]), perform_temp["RES"])
            
    perform_temp["cpuUsage_infer"] = perform_temp["cpuUsage_infer"]/count
    
    performance[num] = perform_temp
    perform_temp[:] = 0

matrix = performance.T
matrix.to_csv(path_to_performance)

