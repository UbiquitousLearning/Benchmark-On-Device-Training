//
//  ResizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class ResizeTest : public MNNTestCase {
public:
    virtual ~ResizeTest() = default;
    virtual bool run() {
        auto input = _Input({1, 2, 2, 1}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);
        auto output                             = _Resize(input, 2.0, 2.0);
        output                                  = _Convert(output, NHWC);
        const std::vector<float> expectedOutput = {-1.0, -1.5, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0,
                                                   3.0,  3.5,  4.0,  4.0,  3.0, 3.5, 4.0, 4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
            MNN_ERROR("ResizeTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim = {1, 4, 4, 1};
        auto gotDim                        = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
            MNN_ERROR("ResizeTest test failed!\n");
            return false;
        }
        return true;
    }
};

class InterpTest : public MNNTestCase {
public:
    virtual ~InterpTest() = default;
    virtual bool run() {
        auto input = _Input({1, 2, 2, 1}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);
       
        //Interp Type:1
        {
            auto output                             = _Interp({input}, 2.0, 2.0, 4, 4, 1, false);
            output                                  = _Convert(output, NHWC);
            const std::vector<float> expectedOutput = {-1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -2.0, -2.0,
                                                        3.0,  3.0,  4.0,  4.0,  3.0, 3.0, 4.0, 4.0};
            auto gotOutput                          = output->readMap<float>();

            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
                MNN_ERROR("InterpType:1 test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpType:1 test failed!\n");
                return false;
            }
        }       
        
        //Interp Type:2
        {
            auto output                             = _Interp({input}, 2.0, 2.0, 4, 4, 2, false);
            output                                  = _Convert(output, NHWC);
            const std::vector<float> expectedOutput = {-1.0, -1.5, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0,
                                                        3.0,  3.5,  4.0,  4.0,  3.0, 3.5, 4.0, 4.0};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
                MNN_ERROR("InterpType:2 test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpType:2 test failed!\n");
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ResizeTest, "op/resize");
MNNTestSuiteRegister(InterpTest, "op/Interp");