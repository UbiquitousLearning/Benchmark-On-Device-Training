# First benchmark suite for on-device training

> The suite is based on MNN, and our unique contribution is placed in [./tools/train/benchmark](./tools/train/benchmark).
## Table of Contents

- [Background](#background)
- [Overview](#overview)
- [Suite Composition](#suite-composition)
  - [NN models](#nn-models)
  - [GPU Training](#gpu-training)
  - [Latency Test](#latency-test)
  - [CPU Configuration](#cpu-configuration)
  - [Thermal Dynamics](#thermal-dynamics)
- [Step-by-step Instructions](#step-by-step-instructions)
  - [root](#root-instruction)
  - [unroot](#unroot-instruction)
- [Example](#example)
- [Roadmap](#roadmap)
- [Related Work](#related-work)
- [Contributors](#contributors)


## Background
Deep learning technique is revolutionizing how edge
devices interact with users or the world, including smartphones and IoT devices. Fueled by the increasingly
powerful on-chip processors, the inference (or prediction) stage of deep learning is known to happen on edge devices without cloud offloading, making a case for low delay and data privacy protection. Beyond inference, the training stage of deep learning is still commonly placed on data centers for its tremendous demand of massive training data and computing resources.

In 2019, alibaba proposed a highly efficient and lightweight deep learning framework: [MNN](https://github.com/alibaba/MNN), which supports inference and training of deep learning models, and has industry leading performance for inference and training on-device. So we develop the first benchmark suite for on-device training based on MNN.

## Overview
Our benchmark suite includes 5 classical NN models, and
can bench CPU/GPU training performance metrics including training latency, energy consumption, memory footprint, hardware utilization, and thermal dynamics. The suite can run on both root and unroot devices.

## Suite Composition

### NN models
Besides from the two NN models that MNN has supported, we add three more models (implemented in [models](../source/models)). Five classical CNN models are tested in our experiments:  LeNet (2 convs, 3.2K parameters), AlexNet (5
convs, 61M parameters), MobileNetv2 (53 convs, 3.4M
parameters), SqueezeNet (18 convs, 411.2K parameters),
and GoogLeNet (22 convs, 6.8M parameters).

### GPU Training
We add essential operations for GPU training using OPENCL and enable preliminary measurement, which means the latency and memory you get are only suggested for reference.

### Latency Test
Our suite can help you test the training/inference latency with different models in a simple way, just need to run [get_data_root.sh](./tools/train/benchmark/get_data_root.sh)/[get_data_unroot.sh](./tools/train/benchmark/get_data_unroot.sh) refer to [instructions](#step-by-step-instructions).

### CPU Configuration
_Because unroot devices have no root for changing cpu configuration, so those function are only available for root devices_

With different parameters for [get_data_root.sh](./tools/train/benchmark/get_data_root.sh), you can change numbers of CPU cores used for training or even specify which core you want to use.

Besides from numbers of CPU, frequency is a important factor affecting training performance as well. So we developed [get_data_root_freq.sh](./tools/train/benchmark/get_data_root_freq.sh) to help you quickly test the difference of training performance under different frequencies.
### Thermal Dynamics
To reach a usable accuracy, the training phase often takes a substantial period of time which may lead to thermal issues and therefore the CPU frequency. So we provide the [freq_temperature monitor](./freq_temperature.sh) for you to know your training device better.

## Step-by-step Instructions

First of all, you need to configure the MNN environment, which can refer to [MNN Instruction](https://www.yuque.com/mnn/cn/build_android).

Then go to the folder of our suite:

```
cd /path/to/root/tools/train/benchmark
```

Whether your device is root or not, you need to push the related dataset to your device before testing. And put the downloaded datasets in the same directory (Dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1IB1-NJgzHSEb7ucgJzM2Gj8QzxpYAjGy?usp=sharing)).

```
./data_prepare.sh /path/to/data/root
```

Next, choose the instructions you need to follow according to the different permissions of your device.
### root
Root device can measure the latency of the model, the influence of different CPU configuration, and the relationship between frequency and temperature.

1. Latency
```
./bench_root.sh ModelName Core Push_model(0/1)
```
2. CPU configuration
```
# numbers of CPU
# Core should be presented in hexadecimal form, .e.g, ff.
./get_data_root.sh numCore Core Device 

# frequency of CPU
./get_data_root_freq.sh Device
```
[get_data_root_freq.sh](./get_data_root_freq.sh) will measure all available frequencies in default. 

Device_Core_${Core} / Device_Freq_${Freq} will be generated in the current directory.


3. Freq-Temperature
```
adb push ./frequency_monitor.sh /data/local/tmp
adb shell
cd /data/local/tmp
chmod 0777 /data/local/tmp/frequency_monitor.sh
/data/local/tmp/frequency_monitor.sh freq_temperature.result 1

# When you think your device is hot enough for frequency down to occur, use ctrl-c in the adb shell, and then run the following code:

python temperature.py
```
batchsize_energy_Device、singleBatch_energy_Device、singleSample_energy_Device willbe generated in the current directory

***TIPS: When you interrupt the measurement ahead of time or the measurement fails, you need to run [clean.sh](./clean.sh) to clear the garbage cache of the last measurement.**

### unroot
```
adb push usage_monitor_unroot.sh /data/local/tmp
adb shell 
cd data/local/tmp
chmod 0777 usage_monitor_unroot.sh
./usage_monitor_unroot.sh usage_monitor.result 1

# Open a new terminal

./get_data_unroot.sh Device
```

## Example
> Device: Meizu 16T (root); numCore: 4 (**f0** i.e. **11110000** i.e. **cpu4、5、6、7**); Frequency: 1.7GHz
```markdown
./data_prepare.sh ~/Desktop/data

./get_data_root.sh 4 f0 MEIZU
# Generate MEIZU_Core_f0/

./get_data_root_freq.sh MEIZU
# Cut down when finish measuring freq 1.7GHz, so only generate MEIZU_Freq_1708800/

adb push ./frequency_monitor.sh /data/local/tmp
adb shell
cd /data/local/tmp
chmod 0777 /data/local/tmp/frequency_monitor.sh
/data/local/tmp/frequency_monitor.sh freq_temperature.result 1

# Run for a while, and open a new terminal to run below
python temperature.py
# Generate batchsize_energy_MEIZU_Freq_1708800.csv singleBatch_energy_MEIZU_Freq_1708800.csv singleSample_energy_MEIZU_Freq_1708800.csv
```

>Device: Xiaomi MI 9 (unroot)

```markdown
adb push usage_monitor_unroot.sh /data/local/tmp
adb shell 
cd data/local/tmp
chmod 0777 usage_monitor_unroot.sh
./usage_monitor_unroot.sh usage_monitor.result 1

# Open a new terminal
./get_data_unroot.sh Device
# Generate XIAOMI/
# Turn back to the old terminal and ctrl-c to stop usage_monitor
```
## Roadmap
In the future, we will add more models for wider measurement. More devices will be measured to find those system parameters influencing on-device training performance and help related scholars choose optimal configuration.

Besides, while being the state-of-the-art training library for edge devices, our experiments show that MNN’s performance is still far from
optima. So we will focus on generating more efficient operators and doing memory optimizations as well.

## Related work
[MNN](https://github.com/alibaba/MNN)

## Contributor
[@caidongqi](https://github.com/caidongqi)
[@xumengwei](https://github.com/xumengwei)
[@wangqipeng](https://github.com/qipengwang)
[@liuyuanqiang](https://github.com/qingyunqu)
