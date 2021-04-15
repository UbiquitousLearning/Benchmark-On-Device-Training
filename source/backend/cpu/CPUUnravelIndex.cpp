//
//  CPUUnravelIndex.cpp
//  MNN
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUUnravelIndex.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {

ErrorCode CPUUnravelIndex::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto indices = inputs[0];
    auto dims    = inputs[1];

    const int elmentSize = indices->elementSize();
    const int dimsSize   = dims->length(0);

    const auto indicesPtr  = indices->host<int32_t>();
    const auto dimsDataPtr = dims->host<int32_t>();
    std::vector<int32_t> mod(dimsSize);
    OpCommonUtils::computeStride(mod.data(), dimsDataPtr, dimsSize);
    auto outputDataPtr = outputs[0]->host<int32_t>();

    std::vector<int> coordinate(dimsSize);
    for (int i = 0; i < elmentSize; ++i) {
        OpCommonUtils::unravelIndexHelper(coordinate, mod, dimsSize, indicesPtr[i]);
        // assign value
        for (int k = 0; k < dimsSize; ++k) {
            outputDataPtr[i + k * elmentSize] = coordinate[k];
        }
    }
    return NO_ERROR;
}

class CPUUnravelIndexCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUUnravelIndex(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUUnravelIndexCreator, OpType_UnravelIndex);

} // namespace MNN
