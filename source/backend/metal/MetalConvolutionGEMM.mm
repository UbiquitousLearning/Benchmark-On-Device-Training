//
//  MetalConvolutionGEMM.mm
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolutionGEMM.hpp"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalConvolution.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

bool MetalConvolutionGEMM::isValid(const Convolution2D *conv, const Tensor *input) {
    auto common = conv->common();
    auto kx = common->kernelX(), ky = common->kernelY();
    if (kx == 1 || ky == 1 || common->group() != 1) {
        return false;
    }
    auto oc = common->outputCount();
    if (oc <= 64) {
        return false;
    }
    auto iw = input->width(), ih = input->height(), ic = input->channel();
    if (iw * ih * ic <= 16384) {
        return false;
    }
    auto sx = common->strideX(), ow = (iw - kx + 1) / sx;
    auto sy = common->strideY(), oh = (ih - ky + 1) / sy;
    if ((iw * ih * ic) / (ow * oh * oc) > 4) {
        return false;
    }

    auto unit = conv->quanParameter() != nullptr ? sizeof(float) : sizeof(metal_float);
    auto iz = UP_DIV(ic, 4), oz = UP_DIV(oc, 4), batch = input->batch();
    return UP_DIV(ow * oh * batch, 4) * kx * ky * iz * 16 * sizeof(metal_float) < (2 << 20) &&  // tmp input
           UP_DIV(ow * oh * batch, 4) * oz * 16 * unit < (2 << 20);                             // tmp output
}

MetalConvolutionGEMM::MetalConvolutionGEMM(Backend *backend, const Tensor *input, const MNN::Op *op)
    : MetalConvolutionCommon(backend, op) {
    loadWeight(op->main_as_Convolution2D());
}

ErrorCode MetalConvolutionGEMM::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // prepare
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    auto ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = output->batch();

    // pad mode support
    int padX = mPadX, padY = mPadY;
    if (mPadMode == PadMode_SAME) {
        int kernelWidthSize = (mKernelX - 1) * mDilateX + 1;
        int kernelHeightSize = (mKernelY - 1) * mDilateY + 1;
        int pw = (ow - 1) * mStrideX + kernelWidthSize - iw;
        int ph = (oh - 1) * mStrideY + kernelHeightSize - ih;
        padX   = pw / 2;
        padY   = ph / 2;
    }

    // create const buffer
    int constants[] = {iw,
                       ih,
                       iw * ih,
                       iz,
                       ow,
                       oh,
                       ow * oh,
                       oz,
                       input->batch(),

                       mKernelX,
                       mKernelY,
                       mKernelX * mKernelY,
                       mStrideX,
                       mStrideY,
                       padX,
                       padY,
                       mDilateX,
                       mDilateY,
                       mActivationType};
    mConstBuffer.reset(sizeof(constants));
    ::memcpy(mConstBuffer.buffer().contents, constants, sizeof(constants));

    // create mat mul const buffer
    int shapes[] = {UP_DIV(ow * oh * ob, 4), oz, mKernelX * mKernelY * iz, 1};
    mShapeBuffer = [context newDeviceBuffer:sizeof(shapes) bytes:shapes access:CPUWriteOnly];

    // accquire space for source & dst
    int is = UP_DIV(ow * oh * ob, 4) * mKernelX * mKernelY * iz * 16 * sizeof(metal_float) / sizeof(uint8_t);
    int os = UP_DIV(ow * oh * ob, 4) * oz * 16 * sizeof(metal_float) / sizeof(uint8_t);
    mTempInput.reset(Tensor::createDevice<uint8_t>(std::vector<int>{is}));
    mTempOutput.reset(Tensor::createDevice<uint8_t>(std::vector<int>{os}));

    if (!backend->onAcquireBuffer(mTempInput.get(), Backend::DYNAMIC) ||
        !backend->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC)) {
        return OUT_OF_MEMORY;
    }
    backend->onReleaseBuffer(mTempInput.get(), Backend::DYNAMIC);
    backend->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    mPipelineGEMM = [context pipelineWithName:@"matmul4x4"];
    mPipelineIm2Col = [context pipelineWithName:@"conv_im2col"];
    mPipelineCol2Im = [context pipelineWithName:@"conv_col2im"];
    NSUInteger gw = UP_DIV(output->width() * output->height() * output->batch(), 4);
    NSUInteger gh = UP_DIV(output->channel(), 4);
    mGemm = [context computeBestGroupAndLocal:mPipelineGEMM threads:{gw, gh, 1}];
    mIm2Col = [context computeBestGroupAndLocal:mPipelineIm2Col threads:{(NSUInteger)ow, (NSUInteger)oh, (NSUInteger)iz*ob}];
    mCol2Im = [context computeBestGroupAndLocal:mPipelineCol2Im threads:{(NSUInteger)ow, (NSUInteger)oh, (NSUInteger)oz*ob}];
    return NO_ERROR;
}

ErrorCode MetalConvolutionGEMM::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return onFloat(inputs[0], outputs[0]);
}

ErrorCode MetalConvolutionGEMM::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto encoder = backend->encoder();

    { // im2col
        [encoder setComputePipelineState:mPipelineIm2Col];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempInput->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mConstBuffer.buffer() offset:0 atIndex:2];
        [encoder dispatchThreadgroups:mIm2Col.first threadsPerThreadgroup:mIm2Col.second];
    }
    { // gemm
        [encoder setComputePipelineState:mPipelineGEMM];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempInput->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempOutput->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mWeight offset:0 atIndex:2];
        [encoder setBuffer:mShapeBuffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:mGemm.first threadsPerThreadgroup:mGemm.second];
    }
    { // col2im
        [encoder setComputePipelineState:mPipelineCol2Im];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempOutput->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mBias offset:0 atIndex:2];
        [encoder setBuffer:mConstBuffer.buffer() offset:0 atIndex:3];
        [encoder dispatchThreadgroups:mCol2Im.first threadsPerThreadgroup:mCol2Im.second];
    }
    return NO_ERROR;
}

template <typename FType, typename TType>
static id<MTLBuffer> weightInBlock(MNNMetalContext *context, int group, int oc, int ic, int kh, int kw,
                                   const FType *src) {
    auto oz     = UP_DIV(oc, 4);
    auto iz     = UP_DIV(ic, 4);
    auto buffer = [context newDeviceBuffer:oz * iz * kw * kh * 16 * sizeof(TType) access:CPUWriteOnly];
    auto dst    = (TType *)buffer.contents;

    for (int o = 0; o < oc; o++) {
        auto zo = o / 4, ro = o % 4;
        auto o_dst = dst + zo * iz * kh * kw * 16 + ro; // o/4 x 4
#pragma clang loop vectorize(enable)
        for (int i = 0; i < ic; i++) {
            auto zi = i / 4, ri = i % 4;
            auto i_dst = o_dst + zi * kh * kw * 16 + ri * 4; // i/4 x 4
#pragma clang loop vectorize(enable)
            for (int h = 0; h < kh; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
                for (int w = 0; w < kw; w++) {
                    // to   [g][o/4][i/4][h][w][16]
                    // from [g][o][i][h][w]
                    i_dst[(h * kw + w) * 16] = *src++;
                }
            }
        }
    }
    return buffer;
}

id<MTLBuffer> MetalConvolutionGEMM::weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    return weightInBlock<float, metal_float>(context, group, oc, ic, kh, kw, src);
}

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
