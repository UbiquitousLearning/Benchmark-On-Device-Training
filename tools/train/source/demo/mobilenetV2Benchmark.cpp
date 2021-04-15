//
//  MobilenetV2Benchmark.cpp
//  MNN
//
//  Created by MNN on 2020/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "MobilenetV2.hpp"
#include "mobilenetV2BenchmarkUtils.hpp"
#include <MNN/expr/NN.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class MobilenetV2TransferModule : public Module {
public:
    MobilenetV2TransferModule(const char* fileName) {
        auto varMap  = Variable::loadMap(fileName);
        auto input   = Variable::getInputAndOutput(varMap).first.begin()->second;
        auto lastVar = varMap["MobilenetV2/Logits/AvgPool"];

        NN::ConvOption option;
        option.channel = {1280, 4};
        mLastConv      = std::shared_ptr<Module>(NN::Conv(option));

        mFix.reset(PipelineModule::extract({input}, {lastVar}, false));

        // Only train last parameter
        registerModel({mLastConv});
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        auto pool   = mFix->forward(inputs[0]);
        auto result = _Softmax(_Reshape(_Convert(mLastConv->forward(pool), NCHW), {0, -1}));
        return {result};
    }
    std::shared_ptr<Module> mFix;
    std::shared_ptr<Module> mLastConv;
};

class MobilenetV2Benchmark : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 6) {
            std::cout << "usage: ./runTrainDemo.out MobilenetV2Benchmark path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt batchsize" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string trainImagesFolder = argv[1];
        std::string trainImagesTxt = argv[2];
        std::string testImagesFolder = argv[3];
        std::string testImagesTxt = argv[4];
        const int batchsize = atoi(argv[5]);

        std::shared_ptr<Module> model(new MobilenetV2);

        MobilenetV2BenchmarkUtils::train(model, 1001, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt, batchsize);

        return 0;
    }
};


DemoUnitSetRegister(MobilenetV2Benchmark, "MobilenetV2Benchmark");
