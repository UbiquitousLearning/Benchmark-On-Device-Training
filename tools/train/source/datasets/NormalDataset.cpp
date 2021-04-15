//
//  NormalDataset.cpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NormalDataset.hpp"
#include <string.h>
#include <fstream>
#include <string>
namespace MNN {
namespace Train {

// referenced from pytorch C++ frontend mnist.cpp
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
const int32_t kTrainSize_224          = 60000/8;
const int32_t kTestSize_224          = 10000/8;
const int32_t kImageMagicNumber_224   = 2051;
const int32_t kTargetMagicNumber_224  = 2049;
const int32_t kImageRows_224          = 28*8;
const int32_t kImageColumns_224       = 28*8;
const char* kTrainImagesFilename_224  = "train-images-idx3-ubyte";
const char* kTrainTargetsFilename_224 = "train-labels-idx1-ubyte";
const char* kTestImagesFilename_224   = "t10k-images-idx3-ubyte";
const char* kTestTargetsFilename_224  = "t10k-labels-idx1-ubyte";

bool check_is_little_endian_224() {
    const uint32_t word = 1;
    return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness_224(uint32_t value) {
    return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) | ((value & 0xff0000u) >> 8u) |
           ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32_224(std::ifstream& stream) {
    static const bool is_little_endian = check_is_little_endian_224();
    uint32_t value;
    stream.read(reinterpret_cast<char*>(&value), sizeof value);
    return is_little_endian ? flip_endianness_224(value) : value;
}

uint32_t expect_int32_224(std::ifstream& stream, uint32_t expected) {
    const auto value = read_int32_224(stream);
    // clang-format off
    // 这里错了，开debug之后过不去
    MNN_ASSERT(value == expected);
    // clang-format on
    return value;
}

std::string join_paths_224(std::string head, const std::string& tail) {
    if (head.back() != '/') {
        head.push_back('/');
    }
    head += tail;
    return head;
}

VARP read_images_224(const std::string& root, bool train) {
    const auto path = join_paths_224(root, train ? kTrainImagesFilename_224 : kTestImagesFilename_224);
    std::ifstream images(path, std::ios::binary);
    if (!images.is_open()) {
        MNN_PRINT("Error opening images file at %s", path.c_str());
        MNN_ASSERT(false);
    }

    const auto count = train ? kTrainSize_224 : kTestSize_224;

    // From http://yann.lecun.com/exdb/mnist/
    expect_int32_224(images, kImageMagicNumber_224);
    expect_int32_224(images, count);
    expect_int32_224(images, kImageRows_224);
    expect_int32_224(images, kImageColumns_224);

    std::vector<int> dims = {count, 3, kImageRows_224, kImageColumns_224};
    int length            = 1;
    for (int i = 0; i < dims.size(); ++i) {
        length *= dims[i];
    }
    auto data = _Input(dims, NCHW, halide_type_of<uint8_t>());
    images.read(reinterpret_cast<char*>(data->writeMap<uint8_t>()), length);
    return data;
}

VARP read_targets_224(const std::string& root, bool train) {
    const auto path = join_paths_224(root, train ? kTrainTargetsFilename_224 : kTestTargetsFilename_224);
    std::ifstream targets(path, std::ios::binary);
    if (!targets.is_open()) {
        MNN_PRINT("Error opening images file at %s", path.c_str());
        MNN_ASSERT(false);
    }

    const auto count = train ? kTrainSize_224 : kTestSize_224;

    expect_int32_224(targets, kTargetMagicNumber_224);
    expect_int32_224(targets, count);

    std::vector<int> dims = {count};
    int length            = 1;
    for (int i = 0; i < dims.size(); ++i) {
        length *= dims[i];
    }
    auto labels = _Input(dims, NCHW, halide_type_of<uint8_t>());
    targets.read(reinterpret_cast<char*>(labels->writeMap<uint8_t>()), length);

    return labels;
}

NormalDataset::NormalDataset(const std::string root, Mode mode)
    : mImages(read_images_224(root, mode == Mode::TRAIN)), mLabels(read_targets_224(root, mode == Mode::TRAIN)) {
    mImagePtr  = mImages->readMap<uint8_t>();
    mLabelsPtr = mLabels->readMap<uint8_t>();
}

Example NormalDataset::get(size_t index) {
    auto data  = _Input({3, kImageRows_224, kImageColumns_224}, NCHW, halide_type_of<uint8_t>());
    auto label = _Input({}, NCHW, halide_type_of<uint8_t>());

    auto dataPtr = mImagePtr + index * kImageRows_224 * kImageColumns_224;
    ::memcpy(data->writeMap<uint8_t>(), dataPtr, kImageRows_224 * kImageColumns_224);

    auto labelPtr = mLabelsPtr + index;
    ::memcpy(label->writeMap<uint8_t>(), labelPtr, 1);

    auto returnIndex = _Const(index);
    // return the index for test
    return {{data, returnIndex}, {label}};
}

size_t NormalDataset::size() {
    return mImages->getInfo()->dim[0];
}

const VARP NormalDataset::images() {
    return mImages;
}

const VARP NormalDataset::labels() {
    return mLabels;
}

DatasetPtr NormalDataset::create(const std::string path, Mode mode) {
    DatasetPtr res;
    res.mDataset.reset(new NormalDataset(path, mode));
    return res;
}
}
}
