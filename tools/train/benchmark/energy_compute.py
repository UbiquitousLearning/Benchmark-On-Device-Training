# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import pandas as pd
import sys
import numpy as np


# %%
# set parameters
para_num = len(sys.argv) # para_num check
if para_num != 2:
    print("usage: python energy_compute.py DeviceName")
    sys.exit(0)

device_name = sys.argv[1]
print(device_name)
net_name = ["Lenet", "GoogLenet", "Squeezenet", "Alexnet", "Mobilenet"]
# net_name = net_name[:4]
batchsizes = [1, 2, 4, 8, 16]


# %%
# 单个freq，不同网络、不同batchsize 训练过程中消耗的能量 p*t p使用所有训练时刻的电压*电流均值（只看usb）减去基准功率（前5条数据的最小值/均值/最大值）t使用end - start ｜stamp
df = pd.DataFrame(columns=net_name, index=batchsizes)

path_to_baseline = "./" + device_name + "/processed_data/Lenet/processed_data_Lenet.csv"
perform = pd.read_csv(path_to_baseline, index_col=0)
baseline_warmup = 10
baseline = int(np.mean(perform.loc[:baseline_warmup]["usb_current"])) * int(np.mean(perform.loc[:baseline_warmup]["usb_voltage"]))# 日常功率的基准值

for net in net_name:
    print(net)
    for batchsize in batchsizes:
        df[net][batchsize] = 0
        path_to_stamp = "./" + device_name + "/train_stamp/" + net + "/" + "train_stamp_" + str(batchsize) + ".result" 
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
        path_to_perform = "./" + device_name + "/processed_data/" + net + "/processed_data_" + net + ".csv"
        perform = pd.read_csv(path_to_perform, index_col=0)
        #print(perform)
        # baseline_warmup = 5
        # baseline = int(np.mean(perform.loc[:baseline_warmup]["usb_current"])) * int(np.mean(perform.loc[:baseline_warmup]["usb_voltage"]))# 日常功率的基准值
        P = []
        for index in perform.index: # 按行对数据进行遍历
            if int(perform.loc[index]["stamp"]) > trainStart and int(perform.loc[index]["stamp"]) < trainEnd:
                # print(perform.loc[index]["usb_current"]*perform.loc[index]["usb_voltage"])
                P.append(perform.loc[index]["usb_current"]*perform.loc[index]["usb_voltage"]-baseline)
        avg_P = np.mean(P)
        #print(avg_P)
        df[net][batchsize] = avg_P * (trainEnd - trainStart)/10**15
df.to_csv("./batchsize_energy_" + device_name + ".csv")
# print(df)
                


# %%
# 测单个sample的energy
df = pd.DataFrame(columns=net_name, index=batchsizes)

data = pd.read_csv("./batchsize_energy_" + device_name + ".csv", index_col=0)

# int warmUp = 2;
# int measureIterations = 32/BatchSize + 5;
# int loops = 1;
# if (NetName == "Lenet") { measureIterations = 1024/BatchSize + 100; } 
for batch in batchsizes:
    for net in net_name:
        if net == "Lenet":
            batchnum = 2 + 1024/batch + 100 + 1
        elif net == "Mobilenet":
            batchnum = 26
        else:
            batchnum = 26
        df[net][batch] = data[net][batch]/(batchnum*batch)

df.to_csv("./singleSample_energy_" + device_name + ".csv")


# %%
# 测单个batch的energy
df = pd.DataFrame(columns=net_name, index=batchsizes)

data = pd.read_csv("./batchsize_energy_" + device_name + ".csv", index_col=0)

for batch in batchsizes:
    for net in net_name:
        if net == "Lenet":
            batchnum = 2 + 1024/batch + 100 + 1
        elif net == "Mobilenet":
            batchnum = 26
        else:
            batchnum = 26
        df[net][batch] = data[net][batch]/(batchnum)

df.to_csv("./singleBatch_energy_" + device_name + ".csv")


