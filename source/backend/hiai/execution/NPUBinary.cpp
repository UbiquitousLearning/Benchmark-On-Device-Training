//
//  NPUBinary.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBinary.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {


void NPUBinary::OpInsert(int binary_type, vector<pair<shared_ptr<ge::Operator>, string>>& ops, string opName, ge::Operator& input0, ge::Operator& input1){

    if(binary_type == BinaryOpOperation_ADD) {
        shared_ptr<ge::op::Add> binary(new ge::op::Add(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x1(input0)
        .set_input_x2(input1);
    } else if(binary_type == BinaryOpOperation_MUL) {
        shared_ptr<ge::op::Mul> binary(new ge::op::Mul(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x(input0)
        .set_input_y(input1);
    } else if(binary_type == BinaryOpOperation_REALDIV) {
        shared_ptr<ge::op::RealDiv> binary(new ge::op::RealDiv(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x1(input0)
        .set_input_x2(input1);
    } else if(binary_type == BinaryOpOperation_SUB) {
        shared_ptr<ge::op::Sub> binary(new ge::op::Sub(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x1(input0)
        .set_input_x2(input1);
    } else if(binary_type == BinaryOpOperation_MINIMUM) {
        shared_ptr<ge::op::Minimum> binary(new ge::op::Minimum(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x1(input0)
        .set_input_x2(input1);
    } else if(binary_type == BinaryOpOperation_MAXIMUM) {
        shared_ptr<ge::op::Maximum> binary(new ge::op::Maximum(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x1(input0)
        .set_input_x2(input1);
    } else if(binary_type == BinaryOpOperation_EQUAL) {
        shared_ptr<ge::op::Equal> binary(new ge::op::Equal(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x1(input0)
        .set_input_x2(input1);
    } else if(binary_type == BinaryOpOperation_LESS_EQUAL) {
        shared_ptr<hiai::op::LessEqual> binary(new hiai::op::LessEqual(opName));
        ops.emplace_back(make_pair(binary, ""));
        (*binary)
        .set_input_x1(input0)
        .set_input_x2(input1);
    }else{
        MNN_ERROR("npu binary not support type : %d \n", binary_type);
        MNN_ASSERT(false);
    }
}

NPUBinary::NPUBinary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    auto opName = mOp->name()->str();
    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    vector<pair<shared_ptr<ge::Operator>, string>> ops;
    auto binary_type = mOp->main_as_BinaryOp()->opType();

    if(!isConst0 && isConst1){
        // 
        auto inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0       = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0        = iops0.back().first;
        auto input1 = inputs[1];
        auto input0 = inputs[1];
        // om input weight const op
        mConst = ge::op::Const(opName + "_w_const");
        {
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();

            auto shape = tensorShapeFormat(input1,inputs[0]);
            ge::TensorDesc fdesc(ge::Shape(shape), ge::FORMAT_NCHW, ge::DT_FLOAT);
            filter->SetTensorDesc(fdesc);
            if (TensorUtils::getDescribe(inputs[1])->dimensionFormat == MNN::MNN_DATA_FORMAT_NCHW) {
                filter->SetData((uint8_t *)input1->host<float>(), input1->elementSize() * sizeof(float));
                mConst.set_attr_value(filter);
            }else{
                vector<float> temp(input1->elementSize(), 0);
                NHWC2NCHW((float*)input1->host<float>(), (float*)temp.data(), shape[0], shape[1], shape[2]*shape[3]);
                filter->SetData((uint8_t *)temp.data(), temp.size() * sizeof(float));
                mConst.set_attr_value(filter);
            }

            filter->SetData((uint8_t *)input1->host<float>(), input1->elementSize() * sizeof(float));
            mConst.set_attr_value(filter);
        }

    }else if(isConst0 && !isConst1){
        // 
        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        // om input weight const op
        mConst = ge::op::Const(opName + "_w_const");
        {
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            auto shape = tensorShapeFormat(input0);
            ge::TensorDesc fdesc(ge::Shape(shape), ge::FORMAT_NCHW, ge::DT_FLOAT);
            filter->SetTensorDesc(fdesc);
            if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN::MNN_DATA_FORMAT_NCHW) {
                filter->SetData((uint8_t *)input0->host<float>(), input0->elementSize() * sizeof(float));
                mConst.set_attr_value(filter);
            }else{
                vector<float> temp(input0->elementSize(), 0);
                NHWC2NCHW((float*)input0->host<float>(), (float*)temp.data(), shape[0], shape[1], shape[2]*shape[3]);
                filter->SetData((uint8_t *)temp.data(), temp.size() * sizeof(float));
                mConst.set_attr_value(filter);
            }
            filter->SetData((uint8_t *)input0->host<float>(), input0->elementSize() * sizeof(float));
            mConst.set_attr_value(filter);
        }
        
    }
}

ErrorCode NPUBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    vector<pair<shared_ptr<ge::Operator>, string>> ops;
    auto binary_type = mOp->main_as_BinaryOp()->opType();

    if(!isConst0 && isConst1){
        // 
        auto inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0       = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0        = iops0.back().first;

        OpInsert(binary_type, ops, opName, *xOp0.get(), mConst);
    }else if(isConst0 && !isConst1){
        // 
        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
       
        OpInsert(binary_type, ops, opName, mConst, *xOp1.get());
        
    }else{

        // 
        auto inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0       = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0        = iops0.back().first;

        
        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.front().first;
        
        OpInsert(binary_type, ops, opName, *xOp0.get(), *xOp1.get());

    }

    auto index = mOp->outputIndexes()->data()[0];
    mNpuBackend->mGrapMap.insert(make_pair(index, ops));
    return NO_ERROR;
}


NPUCreatorRegister<TypedCreator<NPUBinary>> __bianry_op(OpType_BinaryOp);

} // namespace MNN