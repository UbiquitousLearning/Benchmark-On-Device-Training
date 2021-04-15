//
//  mnistBenchmark.cpp
//  MNN
//  对mnistTrain的train和infer的时间进行度量
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "Lenet.hpp"
#include "Alexnet.hpp"
#include "GoogLeNet.hpp"
#include "SqueezeNet.hpp"
#include "MnistBenchmarkUtils.hpp"
#include <MNN/expr/NN.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "module/PipelineModule.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class MnistV2 : public Module {
public:
    MnistV2() {
        NN::ConvOption convOption;
        convOption.kernelSize = {5, 5};
        convOption.channel    = {1, 8};
        convOption.depthwise  = false;
        conv1.reset(NN::Conv(convOption));
        bn.reset(NN::BatchNorm(8));
        convOption.reset();
        convOption.kernelSize = {5, 5};
        convOption.channel    = {8, 8};
        convOption.depthwise  = true;
        conv2.reset(NN::ConvTranspose(convOption));
        convOption.reset();
        convOption.channel    = {512, 100};
        convOption.fusedActivationFunction = NN::Relu6;
        ip1.reset(NN::Conv(convOption));
        convOption.channel    = {100, 10};
        convOption.fusedActivationFunction = NN::None;
        ip2.reset(NN::Conv(convOption));
        registerModel({conv1, bn, conv2, ip1, ip2});
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        VARP x = inputs[0];
        x      = conv1->forward(x);
        x      = bn->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = conv2->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = _Reshape(x, {0, -1, 1, 1});
        //auto info = x->getInfo();
        x      = ip1->forward(x);
        x      = ip2->forward(x);
        //x      = _Convert(x, NCHW);
        x      = _Reshape(x, {0, 1, -1});
        x      = _Softmax(x, 2);
        x      = _Reshape(x, {0, -1});
        return {x};
    }
    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> bn;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
};
class MnistInt8 : public Module {
public:
    MnistInt8(int bits) {
        AUTOTIME;
        NN::ConvOption convOption;
        convOption.kernelSize = {5, 5};
        convOption.channel    = {1, 20};
        conv1.reset(NN::ConvInt8(convOption, bits));
        conv1->setName("conv1");
        convOption.reset();
        convOption.kernelSize = {5, 5};
        convOption.channel    = {20, 20};
        convOption.depthwise  = true;
        conv2.reset(NN::ConvInt8(convOption, bits));
        conv2->setName("conv2");
        convOption.reset();
        convOption.kernelSize = {1, 1};
        convOption.channel    = {320, 500};
        convOption.fusedActivationFunction = NN::Relu6;
        ip1.reset(NN::ConvInt8(convOption, bits));
        ip1->setName("ip1");
        convOption.kernelSize = {1, 1};
        convOption.channel    = {500, 10};
        convOption.fusedActivationFunction = NN::None;
        ip2.reset(NN::ConvInt8(convOption, bits));
        ip2->setName("ip2");
        dropout.reset(NN::Dropout(0.5));
        registerModel({conv1, conv2, ip1, ip2, dropout});
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        VARP x = inputs[0];
        x      = conv1->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = conv2->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = _Reshape(x, {0, -1, 1, 1});
        x      = ip1->forward(x);
        x      = dropout->forward(x);
        x      = ip2->forward(x);
        x      = _Reshape(x, {0, -1});
        x      = _Softmax(x, 1);
        return {x};
    }
    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
    std::shared_ptr<Module> dropout;
};

static void train(std::shared_ptr<Module> model, std::string root, int BatchSize, std::string NetName) {
    MnistBenchmarkUtils::train(model, root, BatchSize, NetName);
}


class MnistBenchmark : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 4) {
            std::cout << "usage: ./runTrainDemo.out MnistBenchmark /path/to/unzipped/mnist/data/  batchsize modelname [depthwise]" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string root = argv[1];
        int batchSize = atoi(argv[2]);
        std::string modelname = argv[3];

        if ("Alexnet" == modelname) {
            std::shared_ptr<Module> model(new Alexnet);
            if (argc >= 5) {
            model.reset(new MnistV2);
            }
            
            train(model, root, batchSize, modelname);
        }

        if ("Squeezenet" == modelname) {
            std::shared_ptr<Module> model(new Squeezenet);
            if (argc >= 5) {
            model.reset(new MnistV2);
            }
            
            train(model, root, batchSize, modelname);
        }

        if ("GoogLenet" == modelname) {
            std::shared_ptr<Module> model(new GoogLenet);
            if (argc >= 5) {
            model.reset(new MnistV2);
            }
            
            train(model, root, batchSize, modelname);
        }
        
        if ("Lenet" == modelname) {
            std::shared_ptr<Module> model(new Lenet);
            if (argc >= 5) {
            model.reset(new MnistV2);
            }
            train(model, root, batchSize, modelname);
        }
        return 0;
    }
};

DemoUnitSetRegister(MnistBenchmark, "MnistBenchmark");
