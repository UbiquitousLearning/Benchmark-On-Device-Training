if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit
fi

outFile=$1
intvl=$2

echo "------------Usage Stats-------------" > $outFile

while true
do	
	export pid=`pgrep Train`
	echo "\n\nNEW DATA $(date +%s%3N)" >> $outFile
	#unroot device cannot read the following data
	echo "usb_current" $(cat /sys/class/power_supply/usb/input_current_now) >> $outFile
	echo "usb_voltage" $(cat /sys/class/power_supply/usb/voltage_now) >> $outFile
	echo "battery_current" $(cat /sys/class/power_supply/battery/current_now) >> $outFile
	echo "battery_voltage" $(cat /sys/class/power_supply/battery/voltage_now) >> $outFile
	echo "\nsystem status" >> $outFile
	cat /proc/stat >> $outFile
	cat /proc/meminfo >> $outFile

	if [ "$pid" -ne 0 ]; then
		echo "\nRunning Code: status of pid $pid" >> $outFile
		top -q -p $pid -n 1 -d 1 >> $outFile
		cat /proc/$pid/stat >> $outFile
		cat /proc/$pid/status >> $outFile
	fi

	if [ "$pid" -eq 0 ]; then # 使得top命令一直运行，这样baseline中就会包含top命令
		echo "\nBaseline" >> $outFile
		top -q -p 1 -n 1 -d 1 >> $outFile
		cat /proc/1/stat >> $outFile
		cat /proc/1/status >> $outFile
	fi

	echo "\nEND" >> $outFile
	usleep $intvl
done