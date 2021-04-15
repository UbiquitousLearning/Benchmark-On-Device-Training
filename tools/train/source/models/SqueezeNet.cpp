//  Squeezenet.cpp
//  MNN
//
//  Created by CDQ on 2021/03/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "SqueezeNet.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;
class _fireMoudle : public Module {
public:
    _fireMoudle(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> conv3;

};

std::shared_ptr<Module> fireMoudle(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3) {
    return std::shared_ptr<Module>(new _fireMoudle(inputChannel, squeeze_1x1, expand_1x1, expand_3x3));
}

_fireMoudle::_fireMoudle(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3) {
    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannel, squeeze_1x1};
    convOption.padMode    = Express::VALID;
    conv1.reset(NN::Conv(convOption));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {squeeze_1x1, expand_1x1};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));

    convOption.kernelSize = {3, 3};
    convOption.channel    = {squeeze_1x1, expand_3x3};
    convOption.padMode    = Express::SAME;
    conv3.reset(NN::Conv(convOption));


    registerModel({conv1, conv2, conv3});
}

std::vector<Express::VARP> _fireMoudle::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv1->forward(x);
    auto y1 = conv2->forward(x);
    auto y2 = conv3->forward(x);
    auto z = _Concat({y1, y2}, 1);
    return {z};
}

Squeezenet::Squeezenet(){
    NN::ConvOption convOption;
    convOption.kernelSize = {7, 7};
    convOption.channel    = {3, 96};
    convOption.stride     = {2, 2};
    convOption.padMode    = Express::SAME;
    conv1.reset(NN::Conv(convOption));
    
    // 第一次写的时候这里都是conv1.reset，导致registerModel的时候失败
    convOption.kernelSize = {1, 1};
    convOption.channel    = {512, 10};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));


    fire1 = fireMoudle(96, 16, 64, 64);
    fire2 = fireMoudle(128, 16, 64, 64);
    fire3 = fireMoudle(128, 32, 128, 128);
    fire4 = fireMoudle(256, 32, 128, 128);
    fire5 = fireMoudle(256, 48, 192, 192);
    fire6 = fireMoudle(384, 48, 192, 192);
    fire7 = fireMoudle(384, 64, 256, 256);
    fire8 = fireMoudle(512, 64, 256, 256);

    registerModel({conv1, conv2, fire1, fire2, fire3, fire4, fire5, fire6,
                   fire7, fire8});
    // registerModel({conv1, conv2, conv3});
}

std::vector<Express::VARP> Squeezenet::onForward(const std::vector<Express::VARP>& inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x      = conv1->forward(x);
    x      = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x      = fire1->forward(x);
    x      = fire2->forward(x);
    x      = fire3->forward(x);
    x      = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x      = fire4->forward(x);
    x      = fire5->forward(x);
    x      = fire6->forward(x);
    x      = fire7->forward(x);
    x      = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x      = fire8->forward(x);
    x      = conv2->forward(x);
    x      = _AvePool(x, {14, 14}, {1, 1}, VALID);
    x      = _Reshape(x, {0, -1});
    return {x};
}

// // fire module in Squeezenet model
// static VARP fireMoudle(VARP x, int inputChannel, int squeeze_1x1,
//                        int expand_1x1, int expand_3x3) {
//     x = _Conv(0.0f, 0.0f, x, {inputChannel, squeeze_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     auto y1 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     auto y2 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_3x3}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
//     return _Concat({y1, y2}, 1); // concat on channel axis (NCHW)
// }

// VARP SqueezenetExpr(int numClass) {
//     auto x = _Input({1, 3, 224, 224}, NC4HW4);
//     x = _Conv(0.0f, 0.0f, x, {3, 96}, {7, 7}, SAME, {2, 2}, {1, 1}, 1);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = fireMoudle(x, 96, 16, 64, 64);
//     x = fireMoudle(x, 128, 16, 64, 64);
//     x = fireMoudle(x, 128, 32, 128, 128);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = fireMoudle(x, 256, 32, 128, 128);
//     x = fireMoudle(x, 256, 48, 192, 192);
//     x = fireMoudle(x, 384, 48, 192, 192);
//     x = fireMoudle(x, 384, 64, 256, 256);
//     x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//     x = fireMoudle(x, 512, 64, 256, 256);
//     x = _Conv(0.0f, 0.0f, x, {512, numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
//     x = _AvePool(x, {14, 14}, {1, 1}, VALID);
//     return x;
// }


} // namespace Model
} // namespace Train
} // namespace MNN
