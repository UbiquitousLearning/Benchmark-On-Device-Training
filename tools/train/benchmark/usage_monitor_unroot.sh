if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit
fi

outFile=$1
intvl=$2

echo Monitoring device stats...
echo "------------Usage Stats-------------" > $outFile

while true
do	
	export pid=`pgrep Train`
	echo "\n\nNEW DATA $(date +%s%3N)" >> $outFile

	if [ "$pid" -ne 0 ]; then
		echo "\nRunning Code: status of pid $pid" >> $outFile
		top -q -p $pid -n 1 -d 1 >> $outFile
		cat /proc/$pid/stat >> $outFile
		cat /proc/$pid/status >> $outFile
	fi

	echo "\nEND" >> $outFile
	usleep $intvl
done