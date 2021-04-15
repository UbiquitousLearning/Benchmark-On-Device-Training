cat ./clean_status/bench_root.sh > bench_root.sh
cat ./clean_status/MnistBenchmarkUtils.cpp > ../source/demo/MnistBenchmarkUtils.cpp
cat ./clean_status/mobilenetV2BenchmarkUtils.cpp > ../source/demo/mobilenetV2BenchmarkUtils.cpp

adb shell < ./cmd.txt

CPU=(0 1 2 3 4 5 6 7)
for num in ${CPU[@]}
do
    adb shell "su -0 chmod 0777 /sys/devices/system/cpu/cpu${num}/cpufreq/scaling_governor"
    adb shell "echo 'schedutil' > /sys/devices/system/cpu/cpu${num}/cpufreq/scaling_governor"
done

echo Clean successfully