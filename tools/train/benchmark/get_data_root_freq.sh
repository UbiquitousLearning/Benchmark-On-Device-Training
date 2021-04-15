if [ "$#" -ne 1 ]; then
    echo "usage: ./get_data_root_freq.sh Device"
    exit
fi

adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies > /data/local/tmp/freq_list.result" 
adb pull /data/local/tmp/freq_list.result ./

freq_list=(`cat freq_list.result`)
echo ${#freq_list[@]}
CPU=(0 1 2 3 4 5 6 7)
NetNames=("Lenet" "GoogLenet" "Squeezenet" "Alexnet" "Mobilenet")

for freq in ${freq_list[@]:16:${#freq_list[@]}} # exclude the lowest 2 cause it's almost impossible for mobile system to be in that status
do
    echo Change freq to ${freq}
    # Change frequency
    for num in ${CPU[@]}
    do
        adb shell "su -0 chmod 0777 /sys/devices/system/cpu/cpu${num}/cpufreq/scaling_governor"
        adb shell "su -0 chmod 0777 /sys/devices/system/cpu/cpu${num}/cpufreq/scaling_setspeed"
        adb shell "echo 'userspace' > /sys/devices/system/cpu/cpu${num}/cpufreq/scaling_governor"
        adb shell "echo $freq > /sys/devices/system/cpu/cpu${num}/cpufreq/scaling_setspeed"
    done

    DEVICE=$1 # replace it with your device name
    DEVICE=${DEVICE}_Freq_${freq}
    sed -i "" "s/DEVICE=YOUR_DEVICE_NAME/DEVICE=${DEVICE}/" ./bench_root.sh
    
    for NetName in ${NetNames[@]}
    do
        ./bench_root.sh ${NetName} ff 1

        python dataprocess.py $DEVICE $NetName

        sleep 120 # cooling device to avoid freq downgrade
    done

    sed -i "" "s/DEVICE=${DEVICE}/DEVICE=YOUR_DEVICE_NAME/" ./bench_root.sh
done
