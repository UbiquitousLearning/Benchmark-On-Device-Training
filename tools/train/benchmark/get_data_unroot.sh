# adb push usage_monitor_unroot.sh /data/local/tmp
# adb shell 
# cd data/local/tmp
# chmod 0777 usage_monitor_unroot.sh
# ./usage_monitor_unroot.sh usage_monitor.result 1

if [ "$#" -ne 1 ]; then
    echo "usage: ./get_data_unroot.sh Device"
    exit
fi

DEVICE=$1 # replace it with your device name

sed -i "" "s/DEVICE=YOUR_DEVICE_NAME/DEVICE=${DEVICE}/" ./bench_unroot.sh

NetNames=("Lenet" "GoogLenet" "Squeezenet" "Alexnet" "Mobilenet")

for NetName in ${NetNames[@]}
do
    ./bench_unroot.sh ${NetName} 1
    adb pull /data/local/tmp/usage_monitor.result ./${DEVICE}/usage_monitor/usage_monitor_${NetName}.result
    python dataprocess_unroot.py $DEVICE $NetName

    sleep 120 # cooling device to avoid freq downgrade

done

# return to default device name 
sed -i "" "s/DEVICE=${DEVICE}/DEVICE=YOUR_DEVICE_NAME/" ./bench_unroot.sh