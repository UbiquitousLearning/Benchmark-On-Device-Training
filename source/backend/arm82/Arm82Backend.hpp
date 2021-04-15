//
//  Arm82Backend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#ifndef Arm82Backend_hpp
#define Arm82Backend_hpp

#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

// armv82's data type default is fp16, so set
// armv82's dataformat: NC8HW8
#define ARMV82_CHANNEL_UNIT 8

typedef __fp16 FLOAT16;

namespace MNN {
class Arm82Backend : public CPUBackend {
public:
    virtual ~Arm82Backend();
    Arm82Backend(const CPURuntime* runtime);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override;

    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    int numberThread() const {
        return threadNumber();
    }
public:
    class Arm82Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addArm82Creator(OpType t, Arm82Creator* ct);
};

#define REGISTER_ARM82_OP_CREATOR(type, creator) \
    void ___##type##__##creator##__() { \
        Arm82Backend::addArm82Creator(type, new creator); \
    }

inline int ARM82TensorElementSizeHelper(const Tensor* t) {
    int size = 1;
    for (int i = 0; i < t->dimensions(); i++) {
        int currentDimSize = t->length(i);
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = UP_DIV(currentDimSize, 8) * 8;
        }
        size *= currentDimSize;
    }
    return size;
}

} // namespace MNN

#endif /* Arm82Backend_hpp */

#endif
