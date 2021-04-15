
//
//  ZeroShapeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "MNN_generated.h"
using namespace MNN::Express;
class ZeroShapeTest : public MNNTestCase {
public:
    virtual ~ZeroShapeTest() = default;
    virtual bool run() {
        auto input = _Input({1, 0, 4, 1}, NHWC);
        input->setName("input");
        auto output    = _Reshape(input, {0, 0, -1});
        auto info      = output->getInfo();
        auto rightDims = std::vector<int>{1, 0, 4};
        if (info->dim[0] != rightDims[0] || info->dim[1] != rightDims[1] || info->dim[2] != rightDims[2]) {
            return false;
        }
        return true;
    }
};
class ZeroShapeTest2 : public MNNTestCase {
public:
    virtual ~ZeroShapeTest2() = default;
    virtual bool run() {
        auto input = _Input({1, -1, 4, 1}, NHWC);
        input->setName("input");
        auto output = _Reshape(input, {0, 0, -1});
        auto info   = output->getInfo();
        input->writeMap<float>();
        auto outputPtr = output->readMap<float>();
        auto rightDims = std::vector<int>{1, -1, 4};
        if (info->dim[0] != rightDims[0] || info->dim[1] != rightDims[1] || info->dim[2] != rightDims[2]) {
            return false;
        }
        if (nullptr != outputPtr) {
            return false;
        }
        return true;
    }
};
class ZeroShapeTest3 : public MNNTestCase {
public:
    virtual bool run() {
        auto input = _Input({1, 0, 4, 1}, NHWC);
        input->setName("input");
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type = MNN::OpType_Unpack;
        op->main.value = new MNN::AxisT;
        op->main.type = MNN::OpParameter_Axis;
        op->main.AsAxis()->axis = 1;
        auto expr = Expr::create(op.get(), {input}, 3);
        auto output = Variable::create(expr, 0);
        auto info   = output->getInfo();
        if (nullptr != info) {
            return false;
        }
        auto sliceOutput = _Split(input, {4}, 2);
        std::vector<int> dstDims = {1, 0, 1, 1};
        for (auto s : sliceOutput) {
            auto info = s->getInfo();
            if (info->dim != dstDims) {
                return false;
            }
            auto ptr = s->readMap<float>();
            if (nullptr != ptr) {
                return false;
            }
        }
        std::vector<int> padds = {0, 0, 1, 0, 0, 0, 0, 0};
        auto paddings = _Const(padds.data(), {2, 4}, NHWC, halide_type_of<int>());
        auto padOutput = _Pad(input, paddings);
        auto padinfo = padOutput->getInfo();
        if (padinfo->dim != std::vector<int>{1, 1, 4, 1}) {
            return false;
        }
        input->writeMap<float>();
        auto ptr = padOutput->readMap<float>();
        for (int i = 0; i < padinfo->size; ++i) {
            if (ptr[i] > 0.000001f) {
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ZeroShapeTest, "expr/zeroshape");
MNNTestSuiteRegister(ZeroShapeTest2, "expr/zeroshape2");
MNNTestSuiteRegister(ZeroShapeTest3, "expr/zeroshape3");
