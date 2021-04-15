//
//  Pipeline.hpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Pipeline_hpp
#define Pipeline_hpp

#include "Schedule.hpp"
#include "core/Execution.hpp"
#include "geometry/GeometryComputer.hpp"

namespace MNN {
struct OperatorInfo::Info {
    std::string name;
    std::string type;
    float flops = 0.0f;
};
class SizeComputer;
/** pipeline. one session may contains multiple pipeline, and one pipeline may contains more than one unit. */
class Pipeline : public NonCopyable {
public:
    Pipeline(std::vector<Schedule::PipelineInfo>&& info, std::shared_ptr<Backend> major,
             std::shared_ptr<Backend> backup, bool allocInput, bool useGeometry);
    ~Pipeline();
    class UnitInfo : public OperatorInfo {
    public:
        UnitInfo()          = default;
        virtual ~UnitInfo() = default;
        void setUp(const Command& cmd, int index);
    };
    void cloneExecution(const std::map<const Op*, std::shared_ptr<Execution>>& cache);
    const std::map<const Op*, std::shared_ptr<Execution>>& getCache() {
        return mOriginExecution;
    }
public:
    /** encode :
       1. compute shape for every op's inputs and outputs;
       2. geometry transform;
       3. copy op, inputs and outputs tensor info to mBuffer
       static_model:  3; dynamic_model: 1,2,3
    */
    ErrorCode encode(bool isStatic = false);
    /** allocMemory: create Execution and alloc memory for every op */
    ErrorCode allocMemory(bool supportDebug = true);
    /** execute this pipline */
    ErrorCode execute();
    ErrorCode executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after);
    std::vector<Schedule::PipelineInfo>& getPipelineInfo();

private:
    std::shared_ptr<Backend> mBackend;
    std::shared_ptr<Backend> mBackupBackend;
    std::vector<std::shared_ptr<Execution>> mExecutions;
    std::vector<UnitInfo> mDebugInfos;
    CommandBuffer mBuffer;
    std::vector<Schedule::PipelineInfo> mInfo;
    std::vector<Tensor*> mMidConstTensors;
    std::vector<Tensor*> mConstTensors;
    bool mAllocInput;
    bool mInit = false;
    std::map<const Op*, std::shared_ptr<Execution>> mOriginExecution;
#ifndef MNN_BUILD_MINI
    GeometryComputer::Context mContext;
    bool mUseGeometry = true;
#endif
};
} // namespace MNN

#endif /* Pipeline_hpp */
