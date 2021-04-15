if [ "$#" -ne 3 ]; then
    echo "usage: ./get_data_root.sh numCore Core Device"
    exit
fi
numCore=$1
Core=$2
DEVICE=$3 # replace it with your device name
DEVICE=${DEVICE}_Core_${Core}

sed -i "" "s/(MNN_FORWARD_CPU, config, 4)/(MNN_FORWARD_CPU, config, ${numCore})/" ../source/demo/MnistBenchmarkUtils.cpp 
sed -i "" "s/(MNN_FORWARD_CPU, config, 4)/(MNN_FORWARD_CPU, config, ${numCore})/" ../source/demo/mobilenetV2BenchmarkUtils.cpp 
sed -i "" "s/DEVICE=YOUR_DEVICE_NAME/DEVICE=${DEVICE}/" ./bench_root.sh

NetNames=("Lenet" "GoogLenet" "Squeezenet" "Alexnet" "Mobilenet")

for NetName in ${NetNames[@]}
do
    ./bench_root.sh ${NetName} $Core 1

    python dataprocess.py $DEVICE $NetName

    sleep 120 # cooling device to avoid freq downgrade

done

# return to 4 cores and default device name 
sed -i "" 's/(MNN_FORWARD_CPU, config, $numCore)/(MNN_FORWARD_CPU, config, 4)/' ../source/demo/MnistBenchmarkUtils.cpp 
sed -i "" 's/(MNN_FORWARD_CPU, config, $numCore)/(MNN_FORWARD_CPU, config, 4)/' ../source/demo/mobilenetV2BenchmarkUtils.cpp 
sed -i "" "s/DEVICE=${DEVICE}/DEVICE=YOUR_DEVICE_NAME/" ./bench_root.sh