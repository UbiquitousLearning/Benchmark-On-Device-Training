
# auto time measurement shell使用说明

`脚本功能`：测试不同batchsize、不同模型，单个batch的train和infer的时间

对应的`脚本文件`是 `./bench_batchsize_android.sh`

运行之前，需要先把两个数据集放到/data/local/tmp目录下

运行脚本会在本文件夹下生成 benchmark_batchsize.txt

文件内容包含了移动端CPU基本信息、运行报错和时间统计

```
运行示例：

./bench_batchsize_android.sh Alexnet 1 2 4 8 -p

其中
第一个参数:模型的名称（可选有 Mobilenet、Alexnet、Lenet、Squeezenet、GoogLenet)
第二个参数：batchsize（可填参数的个数 1到n个）
最后一个参数：可选，加-p表示将build文件push到手机上，在修改过本地文件后需要加此参数
```

## 如果想调整backend，需要修改的地方有

./benchmark_batchsize_android.sh   `line 3`

*Utils.cpp  `exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 4);"`

---

## 新增功能  (2021.3.15)

记录训练/推断过程中各个节点的时间戳

功能已嵌入，直接运行`bench_batchsize_android.sh`即可，运行结束会在`train_stamp`文件夹下得到时间戳文档`train_stamp_*.result`  （*代表batchsize）

---

## 最新版本（2021.3.21）

运行方式不变

对`bench_batchsize_android.sh`进行了优化，每次运行生成的文件会保存在特定的文件夹下（根据设备名、文件类型区分）

```
例：测试设备为samsungS8p，测试网络模型为Lenet，测试BatchSize为1、2、4、8、16、32、64、128、256

运行脚本会在本地生成samsungS8p文件夹，文件夹下有processed_data、train_bench、train_stamp、usage_monitor四个子文件夹，其中：

processed_data: 存放经过python处理过的网络模型数据
train_bench: 存放网络模型训练和推断过程中的log（包含各过程的时间统计）
train_stamp: 存放各网络模型训练和推断过程中的时间戳
usage_monitor: 训练和推断过程中移动设备的各项性能记录

此外，还会生成一个bench_session.result，其中存放了
```

