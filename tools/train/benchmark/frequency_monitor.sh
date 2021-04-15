#!/bin/sh
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit
fi

outFile=$1
intvl=$2

echo Monitoring device frequency and battery temperature...

echo "------------Frequency Stats-------------" > $outFile

while true
do	
	
    echo "\n\nNEW DATA $(date +%s%3N)" >> $outFile
    echo $(dumpsys battery) >> $outFile
    echo "cpu0" $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) >> $outFile
    echo "cpu1" $(cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq) >> $outFile
    echo "cpu2" $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq) >> $outFile
    echo "cpu3" $(cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq) >> $outFile
    echo "cpu4" $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq) >> $outFile
    echo "cpu5" $(cat /sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq) >> $outFile
    echo "cpu6" $(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq) >> $outFile
    echo "cpu7" $(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq) >> $outFile
    echo "\nEND" >> $outFile
    
	sleep $intvl
done