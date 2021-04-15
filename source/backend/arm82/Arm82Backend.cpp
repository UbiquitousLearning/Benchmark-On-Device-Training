//
//  Arm82Backend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include <algorithm>
#include <mutex>

#include "backend/arm82/Arm82Backend.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"

#include "half.hpp"

namespace MNN {

void registerArm82Ops();

static inline std::map<OpType, Arm82Backend::Arm82Creator*>* getArm82CreatorContainer() {
    static std::once_flag fg;
    static std::map<OpType, Arm82Backend::Arm82Creator*>* ret = nullptr;
    std::call_once(fg, [&] { ret = new std::map<OpType, Arm82Backend::Arm82Creator*>; });
    return ret;
}

bool Arm82Backend::addArm82Creator(OpType t, Arm82Creator* ct) {
    auto creatorContainer = getArm82CreatorContainer();
    if (creatorContainer->find(t) == creatorContainer->end()) {
        creatorContainer->insert(std::make_pair(t, ct));
    }
    return true;
}

Arm82Backend::Arm82Backend(const CPURuntime* runtime) : CPUBackend(runtime, MNN_FORWARD_CPU_EXTENSION) {
    // nothing to do
}

Arm82Backend::~Arm82Backend() {
    // nothing to do
}

Execution* Arm82Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const MNN::Op* op) {
    for (auto t : outputs) {
        if (t->getType().code != halide_type_float) {
            return nullptr;
        }
    }
    auto creatorContainer = getArm82CreatorContainer();
    // MNN_PRINT("====> create Execution for type: %s\n", MNN::EnumNameOpType(op->type()));
    auto iter = creatorContainer->find(op->type());

    if (iter == creatorContainer->end()) {
//        MNN_PRINT("[MNNWarning]: ARMV82 don't support type: [%s]\n", MNN::EnumNameOpType(op->type()));
        return nullptr;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (exe == nullptr) {
//        MNN_PRINT("[MNNWarning]: ARMV82 don't support type: [%s]\n", MNN::EnumNameOpType(op->type()));
        return nullptr;
    }
    return exe;
}

static int _getAliginSize(const halide_buffer_t& buffer, MNN_DATA_FORMAT format) {
    // The default data type of input tensor for arm82 backend is FLOAT32.
    // However, Arm82Backend default data type is FLOAT16, so check whether data type is FLOAT32,
    // then divide size by 2
    int size          = sizeof(int16_t);
    const int dimensions = buffer.dimensions;
    for (int i = 0; i < dimensions; i++) {
        int currentDimSize = buffer.dim[i].extent;
        if (format == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = ALIGN_UP8(currentDimSize);
        }
        size *= currentDimSize;
    }
    return size;
}

bool Arm82Backend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
    // arm82 backend tensor data type is fp16 default
    auto tensor = const_cast<Tensor*>(nativeTensor);
    auto& buffer = tensor->buffer();
    if (buffer.type != halide_type_of<float>()) {
        return CPUBackend::onAcquireBuffer(nativeTensor, storageType);
    }
    auto res = allocBuffer(_getAliginSize(buffer, TensorUtils::getDescribe(nativeTensor)->dimensionFormat), (Tensor*)nativeTensor, storageType);
    if (!res) {
        return false;
    }
    // Set mask in device for easy to determine
    buffer.device = 1;
    return true;
}
static void _convertFp16Inside(const halide_buffer_t& ib, const halide_buffer_t& ob, MNN_DATA_FORMAT source, MNN_DATA_FORMAT dest) {
    int area    = 1;
    int channel = 0;
    if (source == dest) {
        ::memcpy(ob.host, ib.host, _getAliginSize(ib, source));
        return;
    }
    if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
        channel = ib.dim[1].extent;
        for (int axis = 2; axis < ib.dimensions; ++axis) {
            area *= ib.dim[axis].extent;
        }
    } else {
        channel = ib.dim[ib.dimensions - 1].extent;
        for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
            area *= ib.dim[axis].extent;
        }
    }

    // external use
    // copy between user and Arm82Backend
    // fp16 fp32 transformation
    const int batch = ib.dim[0].extent;

    if (source == MNN_DATA_FORMAT_NC4HW4 && dest == MNN_DATA_FORMAT_NCHW) {
        const int inbatchStride = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
        const int outBatchStide = channel * area;

        for (int i = 0; i < batch; ++i) {
            MNNNC8HW8TONCHW_NO_TYPE((uint16_t*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                            channel);
        }
        return;
    }

    if (source == MNN_DATA_FORMAT_NCHW && dest == MNN_DATA_FORMAT_NC4HW4) {
        const int inbatchStride = channel * area;
        const int outBatchStide = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
        for (int i = 0; i < batch; ++i) {
            MNNNCHWTONC8HW8_NO_TYPE((uint16_t*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                            channel);
        }
        return;
    }
    MNN_ERROR("Invalide format %d - %d copy for intenal Arm82 Backend\n", source, dest);
}
void Arm82Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto& ib     = srcTensor->buffer();
    auto& ob     = dstTensor->buffer();
    if (ib.type.code != halide_type_float) {
        CPUBackend::onCopyBuffer(srcTensor, dstTensor);
        return;
    }
    auto source = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto srcType = MNN_FORWARD_CPU;
    if (ib.device != 0) {
        srcType = MNN_FORWARD_CPU_EXTENSION;
    }
    auto dstType = MNN_FORWARD_CPU;
    if (ob.device != 0) {
        dstType = MNN_FORWARD_CPU_EXTENSION;
    }
    if (srcType == dstType) {
        if (srcType == MNN_FORWARD_CPU) {
            MNNCPUCopyBuffer(srcTensor, dstTensor);
        } else {
            _convertFp16Inside(ib, ob, source, dest);
        }
        return;
    }
    // Use CPU Copy to turn save format
    std::shared_ptr<Tensor> tempTensor;
    if (source != dest) {
        if (srcType == MNN_FORWARD_CPU) {
            tempTensor.reset(Tensor::create<float>(dstTensor->shape(), nullptr, TensorUtils::getDimType(dstTensor)));
            MNNCPUCopyBuffer(srcTensor, tempTensor.get());
            srcTensor = tempTensor.get();
            source = dest;
        } else {
            tempTensor.reset(Tensor::create<float>(srcTensor->shape(), nullptr, TensorUtils::getDimType(srcTensor)), [dstTensor](void* ptr) {
                auto tempT = (Tensor*)ptr;
                MNNCPUCopyBuffer(tempT, dstTensor);
                delete tempT;
            });
            dstTensor = tempTensor.get();
            dest = source;
        }
    }
    if (source == MNN_DATA_FORMAT_NC4HW4) {
        // NC4HW4 <-> NC8HW8
        int area    = 1;
        int channel = srcTensor->length(1);
        for (int axis = 2; axis < ib.dimensions; ++axis) {
            area *= srcTensor->length(axis);
        }
        const int batch = srcTensor->length(0);
        if (srcType == MNN_FORWARD_CPU) {
            const int outBatchStride = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
            const int inbatchStride = UP_DIV(channel, 4) * area * 4;
            for (int i = 0; i < batch; ++i) {
                MNNNC4HW4TONC8HW8(dstTensor->host<uint16_t>() + outBatchStride * i, srcTensor->host<float>() + inbatchStride * i, area,
                                channel);
            }
        } else {
            const int inbatchStride = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
            const int outBatchStide = UP_DIV(channel, 4) * area * 4;
            for (int i = 0; i < batch; ++i) {
                MNNNC8HW8TONC4HW4(dstTensor->host<float>() + outBatchStide * i, srcTensor->host<uint16_t>() + inbatchStride * i, area,
                                channel);
            }
        }
        return;
    }
    //MNN_PRINT("%d, %d - %d, %d\n", source, srcType, dest, dstType);
    // The format is the same, just convert fp32-fp16
    const int elemenSize = srcTensor->elementSize();
    // copy and quantize/dequantize data
    // cpu -> arm82 copy
    if (srcType == MNN_FORWARD_CPU) {
        const auto src = srcTensor->host<float>();
        auto dst       = dstTensor->host<FLOAT16>();
        MNNQuantizeFP16(dst, src, elemenSize);
        return;
    }
    // arm82 -> cpu copy
    if (srcType == MNN_FORWARD_CPU_EXTENSION) {
        const auto src = srcTensor->host<int16_t>();
        auto dst       = dstTensor->host<float>();
        MNNDequantizeFP16(dst, src, elemenSize);
        return;
    }
    MNN_ERROR("Invalide copy for intenal Arm82 Backend\n");
    return;
}

void registerArm82RuntimeCreator() {
    registerArm82Ops();
};
#ifndef MNN_CODEGEN_REGISTER
static const auto __arm82_global_initializer = []() {
    registerArm82RuntimeCreator();
    return true;
}();
#endif

} // namespace MNN

#endif
