# load raw data and extract their attributes into csv file
import pandas as pd
import sys
import numpy as np

path_to_freq = "./freq_temperature.result"
raw_data = pd.read_table(path_to_freq, header=None, sep=",", engine='python')

attributes= ["stamp", "temperature", "cpu0", "cpu1", "cpu2", "cpu3", "cpu4", "cpu5", "cpu6", "cpu7"]
processed_data = pd.DataFrame(index=attributes)

data = pd.Series(["nan"]*10, attributes)

start_time = int(raw_data[0][1].split()[2])

count = 0
for line in raw_data[0]:
    if "NEW DATA" in line:
        count = count + 1
        data["stamp"] = int(line.split()[2]) - start_time
    if "temperature" in line:
        temp_loc = line.split(" ").index("temperature:")
        #data["temperature"] = 1
        data["temperature"] = line.split(" ")[temp_loc + 1]
    for attribute in attributes[2:]: 
        if attribute in line:
            data[attribute] = line.split()[1]
    if "END" in line:
        processed_data[count] = data
        data[:] = "nan"

matrix = processed_data.T

matrix.to_csv("./freq_temperature.csv")
print("Process successfully!\nData saved in ./freq_temperature.csv")