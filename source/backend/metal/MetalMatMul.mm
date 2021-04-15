//
//  MetalMatMul.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalMatMul.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalMatMul::MetalMatMul(Backend *backend, const MatMul *matmul) : Execution(backend), mConst(static_cast<MetalBackend*>(backend)->runtime()) {
    mTransposeA = matmul->transposeA();
    mTransposeB = matmul->transposeB();
}
ErrorCode MetalMatMul::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    struct matP {
        int size[4];
        int stride[4];
    };
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    matP buffer;
    buffer.size[0] = h;
    buffer.size[1] = e;
    buffer.size[2] = l;
    if (mTransposeA) {
        buffer.stride[0] = 1;
        buffer.stride[1] = e;
    } else {
        buffer.stride[0] = l;
        buffer.stride[1] = 1;
    }
    if (mTransposeB) {
        buffer.stride[2] = l;
        buffer.stride[3] = 1;
    } else {
        buffer.stride[2] = 1;
        buffer.stride[3] = h;
    }
    mConst.reset(sizeof(matP));
    ::memcpy(mConst.buffer().contents, &buffer, sizeof(matP));
    return NO_ERROR;
}

ErrorCode MetalMatMul::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    Tensor* C       = outputs[0];
    auto e = C->length(0);
    auto h = C->length(1);
    
    auto encoder   = backend->encoder();
    if (inputs.size() > 2) {
        auto bandwidth = [context load:@"matmul_bias" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)inputs[2]->deviceId() offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:3];
        [encoder setBuffer:mConst.buffer() offset:0 atIndex:4];
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger)h, (NSUInteger)e, (NSUInteger)1 }
                       bandwidth:bandwidth];
    } else {
        auto bandwidth = [context load:@"matmul" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
        [encoder setBuffer:mConst.buffer() offset:0 atIndex:3];
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger)h, (NSUInteger)e, (NSUInteger)1 }
                       bandwidth:bandwidth];
    }
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalMatMulCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new MetalMatMul(backend, op->main_as_MatMul());
    }
};
REGISTER_METAL_OP_CREATOR(MetalMatMulCreator, OpType_MatMul);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
