//
//  MetalEltwise.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalEltwise.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalEltwise::MetalEltwise(Backend *backend, EltwiseType type) : Execution(backend) {
    auto metal   = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)metal->context();
    mConst             = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    NSString *kernel = nil;
    switch (type) {
        case EltwiseType_PROD:
            kernel = @"eltwise_prod";
            break;
        case EltwiseType_SUM:
            kernel = @"eltwise_add";
            break;
        case EltwiseType_MAXIMUM:
            kernel = @"eltwise_max";
            break;
        default:
            break;
    }
    mPipeline = [context pipelineWithName:kernel];
}
ErrorCode MetalEltwise::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    ((int*)(mConst.contents))[0] = outputs[0]->elementSize();
    auto metal   = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)metal->context();
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outputs[0]->elementSize(), 1, 1)];
    return NO_ERROR;
}

void MetalEltwise::encode(const Tensor *input0, const Tensor *input1, const Tensor *output) {
    auto metal   = static_cast<MetalBackend *>(this->backend());
    auto encoder   = metal->encoder();
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:mConst offset:0 atIndex:3];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

ErrorCode MetalEltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto output = outputs[0];
    encode(inputs[0], inputs[1], output);
    for (int i = 2; i < inputs.size(); i++) {
        encode(inputs[i], output, output);
    }
    return NO_ERROR;
}

class MetalEltwiseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto eltwise = op->main_as_Eltwise();
        return new MetalEltwise(backend, eltwise->type());
    }
};
REGISTER_METAL_OP_CREATOR(MetalEltwiseCreator, OpType_Eltwise);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
