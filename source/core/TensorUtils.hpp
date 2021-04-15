//
//  TensorUtils.hpp
//  MNN
//
//  Created by MNN on 2019/01/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TensorUtils_hpp
#define TensorUtils_hpp

#include <MNN/Tensor.hpp>
#include "Tensor_generated.h"

#ifdef CONSTANT
#undef CONSTANT
#endif // CONSTANT

namespace MNN {
class Backend;
struct TensorArrayAttr {
    // array size is dynamic or not
    bool isDynamicSize = false;
    // elemShape is identical or not
    bool isIdenticalShape = false;
    // the number of element
    uint32_t arraySize = 0;
    // the shape of element
    std::vector<std::vector<int>> elemShape;
};
/** extra tensor info container */
struct Tensor::InsideDescribe {
public:
    /** dimension format */
    MNN_DATA_FORMAT dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
    union {
        /** Serperate memory offset*/
        int offset;

        /** function used to free handle */
        void (*handleFreeFunction)(void*);
    } extra;

    enum MemoryType {
        /** The tensor's memory come from Backend */
        MEMORY_BACKEND = 0,

        /** host memory is owned by tensor or not */
        MEMORY_HOST,

        /** The tensor don't has memory */
        MEMORY_VIRTUAL,

        /** host memory is owned by tensor or not */
        MEMORY_OUTSIDE,

    };
    MemoryType memoryType = MEMORY_BACKEND;
    /** for DEVICE tensor only. backend used to manage tensor's device memory. */
    Backend* backend = nullptr;
    /** for DEVICE tensor only. */
    int useCount = 0;
    enum Usage {
        NORMAL,
        INPUT,
        OUTPUT,
        CONSTANT,
        /** Whether the tensor is a trainable parameter. Trainable parameter should be stored in a different area. */
        TRAINABLE,
    };
    Usage usage = NORMAL;
    struct View {
        int32_t offset = 0;
        int32_t stride[3] = {1, 1, 1};
    };
    struct Region {
        View src;
        View dst;
        int32_t size[3] = {1, 1, 1};
        Tensor* origin;
        // If offset exist, the tensor dimentsion is 2 x N, first N is srcOffsest, second N is dstOffset
        // It need copy N region by the offset tensor set
        Tensor* offset = nullptr;
    };
    std::vector<Region> regions;
    halide_dimension_t dims[MNN_MAX_TENSOR_DIM];
    // TensorArray Attribute
    std::shared_ptr<TensorArrayAttr> tensorArrayAttr;
};
typedef Tensor::InsideDescribe::Usage TensorUsage;

/** tensor utils */
class MNN_PUBLIC TensorUtils {
public:
    /**
     * @brief get extra tensor info.
     * @param tensor    given tensor.
     * @return extra tensor info.
     */
    static Tensor::InsideDescribe* getDescribe(const Tensor* tensor);

    /**
     * @brief copy shape from source tensor to dest tensor.
     * @param source        shape prodiver tensor.
     * @param dest          shape consumer tensor.
     * @param copyFormat    copy data format or not.
     */
    static void copyShape(const Tensor* source, Tensor* dest, bool copyFormat = false);

    /**
     * auto update tensor's strides according to extents and reorder flags.
     * @param tensor    given tensor.
     */
    static void setLinearLayout(Tensor* tensor);

    /**
     * @brief call handle free function to clear handle of tensor.
     * @param tensor    given tensor.
     */
    static void clearHandleData(Tensor* tensor);

    /**
     * @brief compare tensor to expected with tolerance.
     * @param compareTensor comparing tensor.
     * @param toTensor      expected tensor.
     * @param tolerance     tolerable error, any error less than this value will be ignored.
     *                      for integer types, compare with `abs(v1 - v2) > tolerance`;
     *                      for float types, see `overallTolerance`.
     * @param overall       for float types only. compare with `abs(v1 - v2) / max(abs(allExpectValues))` if true,
     *                      `abs(v1 - v2) / abs(v2)` otherwise.
     * @param printsError   print error data or not.
     * @param printsTensors print tensor data or not when meets error.
     * @return equals within tolerance or not.
     */
    static bool compareTensors(const Tensor* compareTensor, const Tensor* toTensor, float tolerance = 0,
                               bool overall = false, bool printsError = true, bool printsTensors = false);

    static void setupTensorInfo(const Tensor* tensor, Tensor* wrapTensor, MNN_DATA_FORMAT mMidFormat);
    static Tensor::InsideDescribe::Region makeFullSlice(Tensor* input);
    static bool regionIsFull(Tensor* input);
    static bool reshapeSlice(Tensor::InsideDescribe::Region& slice, int outside, int inside, int axis);
    static bool fuseRegion(Tensor::InsideDescribe::Region& srcReg, Tensor::InsideDescribe::Region& dstReg);
    static void adjustTensorForCompability(Tensor* t);
    static Tensor::DimensionType getDimType(const Tensor* t);
};
} // namespace MNN

#endif /* TensorDescribe_hpp */
