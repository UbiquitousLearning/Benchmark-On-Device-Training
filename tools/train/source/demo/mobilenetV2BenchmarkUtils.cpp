//
//  MobilenetV2BenchmarkUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "mobilenetV2BenchmarkUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include <MNN/expr/NN.hpp>
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <sys/time.h>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "ImageDataset.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

inline uint64_t MNN_TIME() {
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    uint64_t ret_time = Current.tv_sec * 1000000 + Current.tv_usec;
    return ret_time;
}

void MobilenetV2BenchmarkUtils::train(std::shared_ptr<Module> model, const int numClasses, const int addToLabel,
                                std::string trainImagesFolder, std::string trainImagesTxt,
                                std::string testImagesFolder, std::string testImagesTxt, const int BatchSize,
                                const int trainQuantDelayEpoch, const int quantBits) {
    
    {
        // Load snapshot
        auto para = Variable::load("mobilenetv2.snapshot.mnn");
        model->loadParameters(para);
    }
    system("echo Begin testing : $(date +%s%3N) > /data/local/tmp/train_stamp.result");
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setMomentum(0.9f);
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(0.00004f);

    auto converImagesToFormat  = CV::RGB;
    int resizeHeight           = 224;
    int resizeWidth            = 224;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
    std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
    bool centerOrRandomCrop = false; // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means,cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto trainDataset = ImageDataset::create(trainImagesFolder, trainImagesTxt, datasetConfig.get(), readAllImagesToMemory);
    auto testDataset = ImageDataset::create(testImagesFolder, testImagesTxt, datasetConfig.get(), readAllImagesToMemory);

    // set measurement parameter
    int warmUp = 5;
    int measureIterations = 20;
    
    const int batchSize = BatchSize;
    const int trainBatchSize = batchSize;
    const int trainNumWorkers = 4;
    const int testBatchSize = batchSize;
    const int testNumWorkers = 0;

    auto trainDataLoader = trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);
    auto testDataLoader = testDataset.createLoader(testBatchSize, true, false, testNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();
    const int testIterations = testDataLoader->iterNumber();

    std::vector<float> latency;

    for (int epoch = 0; epoch < 1; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            trainDataLoader->reset();
            model->setIsTraining(true);
            // turn float model to quantize-aware-training model after a delay
            if (epoch == trainQuantDelayEpoch) {
                // turn model to train quant model
                std::static_pointer_cast<PipelineModule>(model)->toTrainQuant(quantBits);
            }
            system("echo Begin training : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");
            for (int i = 0; i < warmUp + measureIterations + 1; i++) {
                if(i == trainIterations) { measureIterations = trainIterations - warmUp - 1; break; }
                auto trainData  = trainDataLoader->next();
                auto example    = trainData[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                                  _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(_Convert(example.first[0], NC4HW4));
                auto loss    = _CrossEntropy(predict, newTarget);
                // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                float rate = 1e-5;
                solver->setLearningRate(rate);
                if (solver->currentStep() % 10 == 0) {
                    std::cout << "train iteration: " << solver->currentStep();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate << std::endl;
                }
                {

                uint64_t mLastResetTime = MNN_TIME();
                solver->step(loss);
                auto lastTime = MNN_TIME();
                auto durations = lastTime - mLastResetTime;
                latency.push_back((float)durations / 1000.0f);
                }
            }

            // 去掉前3个warmup，从第四个latency开始计算，算20个latency的均值
            float latency_count = 0;
            for (int i = warmUp; i < warmUp + measureIterations; i++){
                latency_count += latency[i];
            }
            auto latency_avg = latency_count / float(measureIterations);
            MNN_PRINT("Training latency on MobilenetV2Train is : %f ms (batchsize is %d)\n", latency_avg, BatchSize);
        
            
        }
        system("echo End trainning : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");

        Variable::save(model->parameters(), "mobilenetv2.snapshot.mnn");

        latency.clear();

        int correct = 0;
        int sampleCount = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        exe->gc(Executor::PART);

        AUTOTIME;
        system("echo Begin inferring : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");
        for (int i = 0; i < testIterations; i++) {
            if(i >= testIterations) { break; }
            auto data       = testDataLoader->next();
            auto example    = data[0];
            auto predict    = model->forward(_Convert(example.first[0], NC4HW4));
            predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
            auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
            sampleCount += label->getInfo()->size;
            auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));
            // {
            // AUTOTIME;
            // correct += accu->readMap<int32_t>()[0];
            // }
            if ((i + 1) % 10 == 0) {
                std::cout << "test iteration: " << (i + 1) << " ";
                std::cout << "acc: " << correct << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%";
                std::cout << std::endl;
            }
            {
                uint64_t mLastResetTime = MNN_TIME();

                correct += accu->readMap<int32_t>()[0];

                auto lastTime = MNN_TIME();
                auto durations = lastTime - mLastResetTime;
                // MNN_PRINT("duration is %f ms \n", (float)durations / 1000.0f);
                latency.push_back((float)durations / 1000.0f);
                // MNN_PRINT("第%d个iteration的latency是：%f\n", i, latency.back());
                }
            }

            // 去掉前3个warmup，从第四个latency开始计算，算20个latency的均值
            float latency_count = 0;
            for (int i = warmUp; i < warmUp + measureIterations; i++){
                latency_count += latency[i];
            }
            auto latency_avg = latency_count / float(measureIterations);
            MNN_PRINT("Inferring latency on MobilenetV2Train is : %f ms (batchsize is %d)\n", latency_avg, BatchSize);
        }
        // auto accu = (float)correct / testDataLoader->size();
        // auto accu = (float)correct / usedSize;
        // std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        system("echo End inferring : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");

        {
            auto forwardInput = _Input({batchSize, 3, resizeHeight, resizeWidth}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            Transformer::turnModelToInfer()->onExecute({predict});
            predict->setName("prob");
            std::string modelName = "/data/local/tmp/temp.Mobilenet_" + std::to_string(BatchSize) + ".mnn";
            MNN_PRINT("modeName is %s\n", modelName.c_str());
            Variable::save({predict}, modelName.c_str());
        }

        exe->dumpProfile();
    
}

