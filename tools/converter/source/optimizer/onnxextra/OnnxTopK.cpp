//
//  OnnxTopK.cpp
//  MNNConverter
//
//  Created by MNN on 2020/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <numeric>

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxTopKTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);

        // values, indices = TopK(x, k, axis)
        // Select the top `k` valudes and indices from `x` at axis `axis`.
        //
        // Since tensorflow's TopK only performs at the last axis, so if the `axis` is
        // not the last axis, we had to transpose `x`, do topk with the transposed `x`,
        // and finally transpose the result back.
        auto inputs = expr->inputs();
        int k = 0;
        int axis = 0;
        auto attrs = op->main_as_Extra()->attr();
        MNN_THROW_CHECK(attrs != nullptr, "TopKV's attr is empty");
        for (int i=0; i<attrs->size(); ++i) {
            auto attr = attrs->GetAs<Attribute>(i);
            if (attr->key()->str() == "axis") {
                axis = attr->i();
            } else if (attr->key()->str() == "k") {
                k = attr->i();
            }
        }
        if(inputs.size() == 2) {
            auto ptr = inputs[1]->readMap<int>();
            MNN_THROW_CHECK(ptr != nullptr, "TopKV's k is not const");
            k = ptr[0];
        }

        if (nullptr == inputs[0]->getInfo()) {
            return nullptr;
        }
        int numAxes = inputs[0]->getInfo()->dim.size();
        while (axis < 0) {
            axis += numAxes;
        }
        MNN_ASSERT(axis < numAxes);
        MNN_ASSERT(k <= inputs[0]->getInfo()->dim[axis]);

        std::unique_ptr<TopKV2T> onnxTopKParam(new TopKV2T);
        onnxTopKParam->T      = DataType_DT_FLOAT;
        onnxTopKParam->sorted = false;

        std::unique_ptr<OpT> onnxTopKOp(new OpT);
        onnxTopKOp->name       = op->name()->str();
        onnxTopKOp->type       = OpType_TopKV2;
        onnxTopKOp->main.type  = OpParameter_TopKV2;
        onnxTopKOp->main.value = onnxTopKParam.release();

        EXPRP output = nullptr;
        if (axis != numAxes - 1) {
            std::vector<int> permute(numAxes);
            std::iota(permute.begin(), permute.end(), 0);
            permute[axis]        = numAxes - 1;
            permute[numAxes - 1] = axis;
            VARP transX          = _Transpose(inputs[0], permute);
            auto transY          = Expr::create(onnxTopKOp.get(), {transX, _Scalar<int32_t>(k)}, 2 /*output size*/);
            transY->setName(expr->name());

            // TODO(houjiang): Support tuple expression, and return tuple([values, indices])
            MNN_ERROR("Only the last axis is supported for onnx topk currently.");

            // VARP values = _Transpose(Variable::create(transY, 0), permute);
            // VARP indices = _Transpose(Variable::create(transY, 1), permute);
            // return _Tuple({values, indices});
            return nullptr;
        } else {
            output = Expr::create(onnxTopKOp.get(), {inputs[0], _Scalar<int32_t>(k)}, 2 /*output size*/);
            output->setName(expr->name());
        }
        return output;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("TopK", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxTopKTransformer));
    return true;
}();

} // namespace Express
} // namespace MNN
