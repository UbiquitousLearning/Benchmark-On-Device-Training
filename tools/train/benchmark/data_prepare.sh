if [ "$#" -ne 1 ]; then
    echo "usage: ./data_prepare.sh /path/to/data/root"
    exit
fi

root=$1
ANDROID_DIR=/data/local/tmp

echo pushing data ...
adb push $root/mnist_data $ANDROID_DIR
adb push $root/mobilenet_demo $ANDROID_DIR