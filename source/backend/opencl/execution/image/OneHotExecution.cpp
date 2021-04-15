//
//  OneHotExecution.cpp
//  MNN

#include "backend/opencl/execution/image/OneHotExecution.hpp"

namespace MNN {
namespace OpenCL {

OneHotExecution::OneHotExecution(int axis, Backend *backend): Execution(backend) {
    mAxis = axis;
    auto openCLBackend = static_cast<OpenCLBackend*>(backend);
    auto runtime = openCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    mKernel = runtime->buildKernel("onehot", "onehot", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode OneHotExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto indices        = inputs[0];
    auto depthTensor    = inputs[1];
    auto onValueTensor  = inputs[2];
    auto offValueTensor = inputs[3];
    //indices->print();
    //depthTensor->print();
    //onValueTensor->print();
    //offValueTensor->print();
    //outputs[0]->print();

    int axis = mAxis;
    if (axis < 0) {
        axis += outputs[0]->dimensions();
    }
    int outerSize = 1;
    for (int i = 0; i < axis; ++i) {
        outerSize *= indices->length(i);
    }

    uint32_t idx = 0;
    mKernel.setArg(idx++, openCLImage(indices));
    mKernel.setArg(idx++, openCLImage(depthTensor));
    mKernel.setArg(idx++, outerSize);
    mKernel.setArg(idx++, openCLImage(onValueTensor));
    mKernel.setArg(idx++, openCLImage(offValueTensor));
    mKernel.setArg(idx++, openCLImage(outputs[0]));

    mGlobalWorkSize = {static_cast<uint32_t>(outerSize), 1, 1};
    return NO_ERROR;
}

ErrorCode OneHotExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);

    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us OneHot\n",costTime);
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

    return NO_ERROR;
}

class OneHotCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new OneHotExecution(op->main_as_OneHotParam()->axis(), backend);
    }
};

OpenCLCreatorRegister<OneHotCreator> __onehot_op(OpType_OneHot, IMAGE);

} // namespace OpenCL
} // namespace MNN
