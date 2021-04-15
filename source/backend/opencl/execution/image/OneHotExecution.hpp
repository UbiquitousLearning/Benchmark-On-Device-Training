//
//  OneHotExecution.hpp
//  MNN

#ifndef OneHotExecution_hpp
#define OneHotExecution_hpp

#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class OneHotExecution : public Execution {
public:
    OneHotExecution(int axis, Backend *backend);
    virtual ~OneHotExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize = {1, 1, 1};
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* OneHotExecution_hpp */
