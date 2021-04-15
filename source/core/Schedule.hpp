//
//  Schedule.hpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Schedule_hpp
#define Schedule_hpp

#include <stdio.h>
#include <MNN/Interpreter.hpp>
#include <map>
#include <string>
#include <vector>
#include "core/Backend.hpp"

namespace MNN {

struct Op;
struct Net;

/** net scheduler */
class MNN_PUBLIC Schedule {
public:
    enum Type {
        // Size can be compute seperately
        SEPERATE = 0,
        // When size is fixed, the content is fixed
        CONSTANT = 1,
        // Size can't be compute seperately
        NOT_SEPERATE
    };
    /** pipeline info */
    struct PipelineInfo {
        /** op */
        const Op* op;
        /** input tensors */
        std::vector<Tensor*> inputs;
        /** output tensors */
        std::vector<Tensor*> outputs;
        /** schedule type*/
        Schedule::Type type = Schedule::Type::SEPERATE;
    };

    /** schedule info */
    struct ScheduleInfo {
        /** pipelines with backend info */
        std::vector<std::pair<Backend::Info, std::vector<PipelineInfo>>> pipelineInfo;
        /** input tensors map */
        std::map<std::string, Tensor*> inputTensors;
        /** output tensors map */
        std::map<std::string, Tensor*> outputTensor;
        /** all tensors map */
        std::vector<std::pair<int, std::shared_ptr<Tensor>>> allTensors;
        /** input valid for resize*/
        bool validForResize;
    };

    /**
     * @breif schedule net ops to pipeline with configuration.
     * @param net       given net.
     * @param config    given configuration.
     * @return schedule info.
     */
    static ScheduleInfo schedule(const Net* net, const std::vector<ScheduleConfig>& config);
    static MNNForwardType getApprociateType(const ScheduleConfig& config);
};
} // namespace MNN

#endif /* Schedule_hpp */
