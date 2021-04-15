set -e
ABI="arm64-v8a"
OPENCL="OFF"

if [ "$#" -ne 2 ]; then
    echo "usage: ./bench_unroot.sh ModelName Core Push_model(0/1)"
    exit
fi

ModelName=$1
echo Testing $ModelName
PUSH_MODEL=$2
BatchSize=(1 2 4 8 16)
echo BatchSize: ${BatchSize[*]}
OUT_FILE=train_bench
DEVICE=YOUR_DEVICE_NAME # do not change it here! change it in get_data*.sh
echo Test device is $DEVICE 

BUILD_DIR=build
ANDROID_DIR=/data/local/tmp


function bench_session() {
    adb shell "mkdir -p $ANDROID_DIR/models"
    adb shell "mv $ANDROID_DIR/temp* $ANDROID_DIR/models"
    adb shell "cat /proc/cpuinfo > $ANDROID_DIR/bench_session.result"
    adb shell "echo "
    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/build/benchmark.out $ANDROID_DIR/models/ 10 3 0 >> $ANDROID_DIR/bench_session.result"
    adb pull $ANDROID_DIR/bench_session.result ../$DEVICE/
}

function build_android_bench() {
    mkdir -p $BUILD_DIR
    mkdir -p $DEVICE/train_bench
    mkdir -p $DEVICE/train_stamp/$ModelName
    mkdir -p $DEVICE/usage_monitor
    mkdir -p $DEVICE/processed_data/$ModelName
    cd $BUILD_DIR
    cmake ../../../.. \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_ABI="${ABI}" \
        -DANDROID_STL=c++_static \
        -DMNN_USE_LOGCAT=false \
        -DMNN_OPENCL:BOOL=$OPENCL \
        -DMNN_BUILD_BENCHMARK=ON \
        -DANDROID_NATIVE_API_LEVEL=android-21  \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DMNN_BUILD_TRAIN=ON \
        -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3

    make -j8 
}

function bench_android() {
    build_android_bench
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    
    if [ $PUSH_MODEL -eq 1 ]; then
    echo Pushing models
    adb push ../build $ANDROID_DIR
    fi

    adb shell chmod 0777 $ANDROID_DIR/build/runTrainDemo.out
    adb shell "cat /proc/cpuinfo > $ANDROID_DIR/$OUT_FILE.result"
    adb shell "echo "
    adb shell "echo Build Flags: ABI=$ABI  OpenCL=$OPENCL >> $ANDROID_DIR/$OUT_FILE.result"
    
    for batchsize in ${BatchSize[*]}
    do
        echo "Testing $ModelName with BatchSize $batchsize"

        # benchmark  Alexnet/Lenet/Squeezenet/GoogLenet
        if [ $ModelName != Mobilenet ]; then
        adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/build/runTrainDemo.out MnistBenchmark $ANDROID_DIR/mnist_data $batchsize $ModelName 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/$OUT_FILE.result"
        echo end
        fi

        # benchmark  MnistTrain
        if [ $ModelName == Mobilenet ]; then
        adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/build/runTrainDemo.out MobilenetV2Benchmark $ANDROID_DIR/mobilenet_demo/train_dataset/train_images/ $ANDROID_DIR/mobilenet_demo/train_dataset/train.txt $ANDROID_DIR/mobilenet_demo/test_dataset/test_images/ $ANDROID_DIR/mobilenet_demo/test_dataset/test.txt $batchsize 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/$OUT_FILE.result"
        echo end
        fi

        adb pull $ANDROID_DIR/train_stamp.result ../
        mv -f ../train_stamp.result ../$DEVICE/train_stamp/$ModelName/train_stamp_${batchsize}.result
    done

    echo start bench_session
    bench_session

    adb pull $ANDROID_DIR/$OUT_FILE.result ../$DEVICE/train_bench/${OUT_FILE}_${ModelName}.result
}


bench_android
