//
//  OneHotBufExecution.hpp
//  MNN

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef OneHotBufExecution_hpp
#define OneHotBufExecution_hpp

#include "backend/opencl/core/OpenCLBackend.hpp"
#include "core/Execution.hpp"
#include <vector>

namespace MNN {
namespace OpenCL {

class OneHotBufExecution : public Execution {
public:
    OneHotBufExecution(int axis, Backend *backend);
    virtual ~OneHotBufExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

private:
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize = {1, 1, 1};
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* OneHotBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */