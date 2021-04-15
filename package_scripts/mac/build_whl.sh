set -e

usage() {
    echo "Usage: $0 -o path -v python_versions [-b]"
    echo -e "\t-o package files output directory"
    echo -e "\t-p python versions in pyenv"
    echo -e "\t-v MNN dist version"
    echo -e "\t-b opencl backend"
    exit 1
}

while getopts "o:p:v:b" opt; do
  case "$opt" in
    o ) path=$OPTARG ;;
    p ) IFS="," read -a python_versions <<< $OPTARG ;;
    v ) mnn_version=$OPTARG ;;
    b ) opencl=true ;;
    * ) usage ;;
  esac
done

./schema/generate.sh
rm -rf $path && mkdir -p $path
PACKAGE_PATH=$(realpath $path)

CMAKE_ARGS="-DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF -DMNN_EXPR_SHAPE_EAGER=ON -DMNN_TRAIN_DEBUG=ON"
if [ ! -z $opencl ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DMNN_OPENCL=ON"
fi

rm -rf pymnn_build && mkdir pymnn_build
pushd pymnn_build
cmake $CMAKE_ARGS .. && make MNN MNNTrain MNNConvert -j8
popd

pushd pymnn/pip_package
rm -rf build && mkdir build
rm -rf dist && mkdir dist
for env in $python_versions; do
    pyenv global $env
    python build_wheel.py --version $mnn_version
done
cp dist/* $PACKAGE_PATH

popd
