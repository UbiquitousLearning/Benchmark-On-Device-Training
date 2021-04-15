//
//  Session.hpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Session_hpp
#define Session_hpp

#include <MNN/Tensor.hpp>
#include <map>
#include <memory>
#include <vector>
#include "Pipeline.hpp"
#include "Schedule.hpp"
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "shape/SizeComputer.hpp"

namespace MNN {
struct Net;
/** infer unit. multiple sessions could share one net. */
class MNN_PUBLIC Session {
public:
    Session(Schedule::ScheduleInfo&& info, Interpreter::SessionMode callBackMode, Interpreter::SessionMode inputMode,
            RuntimeInfo&& runtime);
    ~Session();

public:
    /**
     * @brief infer.
     * @return result code.
     */
    ErrorCode run() const;
    /**
     * @brief infer with callbacks and sync option.
     * @param enterCallback callback before each op.
     * @param exitCallback  callback after each op.
     * @param sync          wait until all ops done before return or not.
     * @return result code.
     */
    ErrorCode runWithCallBack(const TensorCallBackWithInfo& enterCallback, const TensorCallBackWithInfo& exitCallback,
                              bool sync = false) const;

    bool getInfo(Interpreter::SessionInfoCode code, void* ptr) const;

    void cloneExecution(const std::map<const Op*, std::shared_ptr<Execution>>& cache, int pipelineIndex);
    const std::map<const Op*, std::shared_ptr<Execution>>& getExecution(int pipelineIndex);
public:
    /**
     * @brief resize tensors and buffers responding to input changes.
     * @return result code.
     */
    ErrorCode resize(bool isStatic = false);
    /**
     * @brief check if needs resize.
     * @return needs resize or not.
     */
    bool getNeedResize() const {
        return mNeedResize;
    }
    /**
     * @brief set if needs resize.
     * @param flag  needs resize or not.
     */
    void setNeedResize(bool flag = true) {
        mNeedResize = flag;
    }

public:
    /**
     * @brief get backend that create the tensor.
     * @param tensor    given tensor.
     * @return backend that create the tensor, NULL if the tensor is created by default backend (CPU backend).
     */
    const Backend* getBackEnd(const Tensor* tensor) const;

    /**
     * @brief get input tensor for given op name.
     * @param name given op name. if NULL, return first input tensor.
     * @return input tensor if found, NULL otherwise.
     */
    Tensor* getInput(const char* name) const;

    /**
     * @brief get output tensor for given op name.
     * @param name given op name. if NULL, return first output tensor.
     * @return output tensor if found, NULL otherwise.
     */
    Tensor* getOutput(const char* name) const;

    /**
     * @brief get output tensors map.
     * @return get output tensors map.
     */
    const std::map<std::string, Tensor*>& getOutputAll() const;
    const std::map<std::string, Tensor*>& getInputAll() const;

    /**
     * @brief check session is valid or not.
     * @return session is valid or not.
     */
    inline bool valid() const {
        return mValid;
    }

    /**
     * @brief update the session's const value to origin model's const blob.
     * @return errorcode
     */
    ErrorCode updateToModel(Net* net) const;

    bool loadCache(const void* buffer, size_t size);
    std::pair<const void*, size_t> getCache();

protected:
    const std::vector<std::shared_ptr<Pipeline>>& getPipelines() const {
        return this->mPipelines;
    }

private:
    void _clearCache();
    void _setUpTensorInfo(const Schedule::ScheduleInfo& info);

private:
    RuntimeInfo mRuntime;
    std::vector<std::shared_ptr<Pipeline>> mPipelines;
    std::vector<std::pair<int, std::shared_ptr<Tensor>>> mTensors;
    std::map<std::string, Tensor*> mInputs;
    std::map<std::string, Tensor*> mOutputs;
    bool mNeedResize = true;
    bool mValid      = true;
    Interpreter::SessionMode mCallBackMode;
};
} // namespace MNN

#endif /* Session_hpp */
