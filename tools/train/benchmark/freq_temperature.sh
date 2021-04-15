adb push ./frequency_monitor.sh /data/local/tmp
adb shell
cd /data/local/tmp
chmod 0777 /data/local/tmp/frequency_monitor.sh
/data/local/tmp/frequency_monitor.sh freq_temperature.result 1

python temperature.py

