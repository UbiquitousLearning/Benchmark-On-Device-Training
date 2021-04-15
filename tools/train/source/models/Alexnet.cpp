//
//  Alexnet.cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Alexnet.hpp"
#include <MNN/expr/NN.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {

Alexnet:: Alexnet() {
    NN::ConvOption convOption;
    convOption.kernelSize = {11, 11};
    convOption.channel    = {3, 96};
    convOption.padMode = VALID;
    convOption.stride    = {4, 4};
    conv1.reset(NN::Conv(convOption));
    convOption.reset();
    convOption.kernelSize = {5, 5};
    convOption.channel    = {96, 256};
    convOption.padMode = SAME;
    conv2.reset(NN::Conv(convOption));
    convOption.reset();
    convOption.kernelSize = {3, 3};
    convOption.channel    = {256, 384};
    convOption.padMode = SAME;
    conv3.reset(NN::Conv(convOption));
    convOption.reset();
    convOption.kernelSize = {3, 3};
    convOption.channel    = {384, 384};
    convOption.padMode = SAME;
    conv4.reset(NN::Conv(convOption));
    convOption.reset();
    convOption.kernelSize = {3, 3};
    convOption.channel    = {384, 256};
    convOption.padMode = SAME;
    conv5.reset(NN::Conv(convOption));

    ip1.reset(NN::Linear(256*5*5, 4096));
    ip2.reset(NN::Linear(4096, 10));

    dropout1.reset(NN::Dropout(0.5));
    dropout2.reset(NN::Dropout(0.5));
    registerModel({conv1, conv2, conv3, conv4, conv5, ip1, ip2, dropout1,dropout2});
}

std::vector<Express::VARP> Alexnet::onForward(const std::vector<Express::VARP>& inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x      = conv1->forward(x); //54x54
    x      = _Relu(x);
    x      = _MaxPool(x, {3, 3}, {2, 2}); //26x26
    x      = conv2->forward(x); //26x26
    x      = _Relu(x);
    x      = _MaxPool(x, {3, 3}, {2, 2}); //12x12
    x      = conv3->forward(x);
    x      = _Relu(x);
    x      = conv4->forward(x);
    x      = _Relu(x);
    x      = conv5->forward(x);
    x      = _MaxPool(x, {3, 3}, {2, 2}); //5x5
    x      = _Reshape(x, {0, -1});
    x      = ip1->forward(x);
    x      = _Relu(x);
    x      = dropout1->forward(x);
    x      = _Relu(x);
    x      = dropout2->forward(x);
    x      = ip2->forward(x);
    x      = _Softmax(x, 1);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN
