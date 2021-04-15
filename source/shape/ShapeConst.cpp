//
//  ShapeConst.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class ConstComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        //MNN_ASSERT(0 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // copy dims
        auto output    = outputs[0];
        auto parameter = op->main_as_Blob();

        output->buffer().dimensions = parameter->dims() ? parameter->dims()->size() : 0;
        for (int i = 0; i < output->buffer().dimensions; i++) {
            output->buffer().dim[i].extent = parameter->dims()->Get(i);
        }
        if (parameter->dataType() == DataType_DT_HALF) {
            output->setType(DataType_DT_FLOAT);
        } else {
            output->setType(parameter->dataType());
        }
        TensorUtils::getDescribe(output)->dimensionFormat = parameter->dataFormat();

        return true;
    }
};

REGISTER_SHAPE(ConstComputer, OpType_Const);
REGISTER_SHAPE(ConstComputer, OpType_TrainableParam);

} // namespace MNN
