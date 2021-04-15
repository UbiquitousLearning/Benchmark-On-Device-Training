//
//  InitNet.cpp
//  MNN
//
//  Created by MNN on 2018/09/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "InitNet.hpp"
#include "core/TensorUtils.hpp"
#include <unordered_map>
namespace MNN {

bool initTensors(std::vector<std::shared_ptr<Tensor>>& tensors, const Net* net) {
    bool valid = true;
    for (int i = 0; i < tensors.size(); ++i) {
        tensors[i].reset(new Tensor(4)); // NCHW, TODO
        tensors[i]->setType(DataType_DT_FLOAT);
    }
    // Set Input Tensor, if the type of input is not the same with ExtraTensorDescribe, use input parameter
    for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
        auto op = net->oplists()->GetAs<Op>(opIndex);
        if (OpType_Input == op->type()) {
            MNN_ASSERT(nullptr != op->outputIndexes());
            MNN_ASSERT(op->outputIndexes()->size() == 1);
            auto index      = op->outputIndexes()->data()[0];
            auto tensor     = tensors[index].get();
            auto& tb        = tensor->buffer();
            auto inputParam = op->main_as_Input();
            if (auto idims = inputParam->dims()) {
                for (int i = 0; i < idims->size(); ++i) {
                    int extent = idims->data()[i];
                    // dim-0 is batch(when input batch is -1, set it to be 1, ignore other dim)
                    if (i == 0 && extent == -1) {
                        extent = 1;
                    }
                    if (extent < 0) {
                        valid = false;
                    }
                    tb.dim[i].extent = extent;
                }
                tb.dimensions = idims->size();
            } else {
                tb.dimensions = 0;
            }
            tensor->setType(inputParam->dtype());
            TensorUtils::getDescribe(tensor)->dimensionFormat = inputParam->dformat();
        }
    }
    return valid;
}
void initPipelineInfosFromOps(std::vector<Schedule::PipelineInfo>& infos, std::vector<const Op*>& ops, const std::vector<std::shared_ptr<Tensor>>& allTensors) {
    for (const Op* op : ops) {
        Schedule::PipelineInfo opInfo;
        opInfo.op = op;
        if (nullptr != op->outputIndexes()) {
            auto data = op->outputIndexes()->data();
            for (int j = 0; j < op->outputIndexes()->size(); ++j) {
                opInfo.outputs.push_back(allTensors[data[j]].get());
            }
        }
        if (nullptr != op->inputIndexes()) {
            auto data = op->inputIndexes()->data();
            for (int j = 0; j < op->inputIndexes()->size(); ++j) {
                opInfo.inputs.push_back(allTensors[data[j]].get());
            }
        }
        infos.emplace_back(std::move(opInfo));
    }
}

void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors, const std::vector<const Op*>& ops, bool isStatic) {
    std::set<int> inputIndexes;
    std::set<int> outputIndexes;
    // 0. deal virtual tensor for static model:
    // when : A (Any_Op) -----> B (Raster_Op)
    // the tensor will be like below:
    //      A_outputs : a_tensor
    //      B_inputs  : b_tensor (virtual)
    //      b_tensor.describe.origin = a_tensor_ptr
    // b_tensor is not a InputTensot, a_tensor is not a OutputTensor
    // so add b_tensor to OutputIndexes, a_tensor to InputIndexes.
    if (isStatic) {
        std::unordered_map<Tensor*, int> tensorMap;
        for (int index = 0; index < allTensors.size(); index++) {
            tensorMap.insert(std::make_pair(allTensors[index].get(), index));
        }
        for (int index = 0; index < allTensors.size(); index++) {
            auto des = TensorUtils::getDescribe(allTensors[index].get());
            for (int i = 0; i < des->regions.size(); i++) {
                outputIndexes.insert(index);
                MNN_ASSERT(tensorMap.find(des->regions[i].origin) != tensorMap.end());
                int x = tensorMap[des->regions[i].origin];
                inputIndexes.insert(x);
            }
        }
    }
    // 1. insert all output/input index in outputIndexes/inputIndexes
    for (auto op : ops) {
        if (nullptr != op->outputIndexes()) {
            auto data = op->outputIndexes()->data();
            for (int j = 0; j < op->outputIndexes()->size(); ++j) {
                outputIndexes.insert(data[j]);
            }
        }
        if (nullptr != op->inputIndexes()) {
            auto data = op->inputIndexes()->data();
            for (int j = 0; j < op->inputIndexes()->size(); ++j) {
                inputIndexes.insert(data[j]);
            }
        }
        MNN_ASSERT(OpType_Input != op->type());
    }
    // 2. the index in outputIndexes/inputIndexed but not in inputIndexes/outputIndexes is output/input
    std::set<int> input;
    std::set<int> output;
    std::set_difference(outputIndexes.begin(), outputIndexes.end(), inputIndexes.begin(), inputIndexes.end(),
                        std::inserter(output, output.begin()));
    std::set_difference(inputIndexes.begin(), inputIndexes.end(), outputIndexes.begin(), outputIndexes.end(),
                        std::inserter(input, input.begin()));
    // 3. set usage for Tensor by index
    for (auto index : input) {
        if (TensorUtils::getDescribe(allTensors[index].get())->usage == TensorUsage::CONSTANT) {
            continue;
        }
        //MNN_PRINT("%d - %p: input\n", index, allTensors[index].get());
        TensorUtils::getDescribe(allTensors[index].get())->usage = TensorUsage::INPUT;
    }
    for (auto index : output) {
        TensorUtils::getDescribe(allTensors[index].get())->usage = TensorUsage::OUTPUT;
    }
}

void initPipelineInfosFromNet(std::vector<Schedule::PipelineInfo>& infos, const Net* net, std::vector<std::shared_ptr<Tensor>>& allTensors) {
    std::vector<const Op*> ops;
    for (int i = 0; i < net->oplists()->size(); i++) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->type() == OpType_Input) {
            continue;
        }
        ops.push_back(op);
    }
    initPipelineInfosFromOps(infos, ops, allTensors);
    setInputOutputForOps(allTensors, ops);
}
} // namespace MNN
