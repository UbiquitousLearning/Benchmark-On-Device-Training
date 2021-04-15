//
//  convertToStaticModel.cpp
//  MNN
//
//  Created by wangzhaode on 2020/9/3.
//
#include <fstream>

#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "utils/InitNet.hpp"
#include "core/Command.hpp"
#include "shape/SizeComputer.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
using namespace MNN;

#define SET_TYPE(TYPE, type) \
if (tensor->getType() == halide_type_of<type##_t>()) {\
blob->dataType = DataType_DT_##TYPE;

#define CONSTANT_COPY(TYPE, type) \
SET_TYPE(TYPE, type)\
for (int i = 0; i < tensor->elementSize(); i++) {\
blob->type##s.push_back(tensor->host<type##_t>()[i]);\
}\
}

void genStaticModel(CommandBuffer buffer, const std::string& modelName, std::map<Tensor*, std::string>& tensorNames) {
    printf("gen Static Model ... \n");
    std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
    netT->usage = Usage_INFERENCE_STATIC;
    std::map<Tensor*, int> tensorMap;
    // add Tensors to netT
    for (auto& iter : buffer.command) {
        std::function<void(Tensor*)> insertTensor = [&](Tensor* t) {
            if (tensorMap.find(t) == tensorMap.end()) {
                auto des = TensorUtils::getDescribe(t);
                if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                    for (auto reg : des->regions) {
                        insertTensor(reg.origin);
                    }
                }
                int index = static_cast<int>(tensorMap.size());
                tensorMap.insert(std::make_pair(t, index));
                std::string tensorName = "ExtraTensor_" + std::to_string(index);
                if (tensorNames.find(t) != tensorNames.end()) {
                    tensorName = tensorNames[t];
                }
                netT->tensorName.push_back(tensorName);
            }
        };
        for (auto& t : iter.inputs) {
            insertTensor(t);
        }
        for (auto& t : iter.outputs) {
            insertTensor(t);
        }
    }
    // add tensors' describe to netT
    for (auto tensorPair : tensorMap) {
        auto tensor = tensorPair.first;
        auto index = tensorPair.second;
        auto des = TensorUtils::getDescribe(tensor);
        if (des->usage == Tensor::InsideDescribe::CONSTANT) {
            std::unique_ptr<OpT> op(new OpT);
            op->type = OpType_Const;
            auto blob = new BlobT;
            op->main.type = OpParameter_Blob;
            op->main.value = blob;
            blob->dataFormat = des->dimensionFormat;
            for (int d = 0; d < tensor->dimensions();d++) {
                blob->dims.push_back(tensor->buffer().dim[d].extent);
            }
            if (tensor->getType() == halide_type_of<float>()) {
                blob->dataType = DataType_DT_FLOAT;
                for (int i = 0; i < tensor->elementSize(); i++) {
                    blob->float32s.push_back(tensor->host<float>()[i]);
                }
            } else {
                CONSTANT_COPY(INT8, int8);
                CONSTANT_COPY(UINT8, uint8);
                CONSTANT_COPY(INT32, int32)
                CONSTANT_COPY(INT64, int64);
            }
            op->outputIndexes.push_back(index);
            netT->oplists.emplace_back(std::move(op));
        }
        auto describe = std::unique_ptr<MNN::TensorDescribeT>(new MNN::TensorDescribeT);
        describe->index = index;
        describe->blob = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
        auto& blob = describe->blob;
        blob->dataFormat = des->dimensionFormat;
        if (tensor->getType() == halide_type_of<float>()) {
            blob->dataType = DataType_DT_FLOAT;
        } else {
            SET_TYPE(INT8, int8)}
            SET_TYPE(UINT8, uint8)}
            SET_TYPE(INT32, int32)}
            SET_TYPE(INT64, int64)}
        }
        for (int d = 0; d < tensor->dimensions();d++) {
            describe->blob->dims.push_back(tensor->buffer().dim[d].extent);
        }
        if (tensor->dimensions() == 0) {
            describe->blob->dims.push_back(1);
        }
        if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
            for (auto& reg : des->regions) {
                auto regionT = std::unique_ptr<MNN::RegionT>(new MNN::RegionT);
                regionT->src = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
                regionT->dst = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
                regionT->src->offset = reg.src.offset;
                regionT->dst->offset = reg.dst.offset;
                for (int s = 0; s < 3; s++) {
                    regionT->src->stride.push_back(reg.src.stride[s]);
                    regionT->dst->stride.push_back(reg.dst.stride[s]);
                    regionT->size.push_back(reg.size[s]);
                }
                regionT->origin = tensorMap[reg.origin];
                describe->regions.emplace_back(std::move(regionT));
            }
        }
        netT->extraTensorDescribe.emplace_back(std::move(describe));
    }
    // add op to netT
    int idx = 0;
    for (auto& iter : buffer.command) {
        if (!iter.buffer.empty()) {
            iter.op = flatbuffers::GetMutableRoot<Op>((void*)iter.buffer.data());
        }
        auto opt = iter.op->UnPack();
        if (opt->name.size() <= 0) {
            opt->name = std::string("Geometry_") + MNN::EnumNameOpType(opt->type) + std::to_string(idx++);
        }
        opt->inputIndexes.resize(iter.inputs.size());
        opt->outputIndexes.resize(iter.outputs.size());
        for (int i = 0; i < iter.outputs.size(); i++) {
            opt->outputIndexes[i] = tensorMap[iter.outputs[i]];
        }
        for (int i = 0; i < iter.inputs.size(); i++) {
            opt->inputIndexes[i] = tensorMap[iter.inputs[i]];
        }
        netT->oplists.emplace_back(std::move(opt));
    }
    // write netT to file
    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    int sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();
    std::ofstream output(modelName, std::ofstream::binary);
    output.write((const char*)bufferOutput, sizeOutput);
}

void converToStaticModel(const Net* net, std::map<std::string,std::vector<int>>& inputConfig, std::string mnnFile) {
    std::vector<std::shared_ptr<Tensor>> allTensors;
    allTensors.resize(net->tensorName()->size());
    initTensors(allTensors, net);
    // set tensors' shape by inputConfig
    for (int i = 0; i < allTensors.size(); i++) {
        auto name = net->tensorName()->GetAsString(i)->str();
        if (inputConfig.find(name) != inputConfig.end()) {
            auto& dims = inputConfig[name];
            for (int j = 0; j < dims.size(); j++) {
                allTensors[i]->buffer().dim[j].extent = dims[j];
            }
        }
    }
    std::vector<Schedule::PipelineInfo> infos;
    initPipelineInfosFromNet(infos, net, allTensors);
#ifdef MNN_BUILD_MINI
    // if MNN-MIN, mnn lib wouldnt init SizeComputer and GeometryComputer
    // init them for shape compute and geometry transform
    SizeComputerSuite::init();
    GeometryComputer::init();
#endif
    // set a backend and context to run resize
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_High;
    config.backendConfig = &backendConfig;
    Backend::Info compute;
    compute.type = config.type;
    compute.numThread = config.numThread;
    compute.user = config.backendConfig;
    const RuntimeCreator* runtimeCreator(MNNGetExtraRuntimeCreator(compute.type));
    std::unique_ptr<Runtime> runtime(runtimeCreator->onCreate(compute));
    std::shared_ptr<Backend> backend(runtime->onCreate());
    GeometryComputer::Context ctx(backend, true);
    CommandBuffer buffer;
    // resize the session's info and store to buffer
    std::vector<Tensor*> constTensors;
    std::vector<Tensor*> midConstTensors;
    GeometryComputerUtils::buildConstantTensors(infos, backend, true, constTensors, midConstTensors);
    GeometryComputerUtils::shapeComputeAndGeometryTransform(infos, buffer, ctx, backend);
    std::map<Tensor*, std::string> tensorName;
    for (int i = 0; i < net->tensorName()->size(); i++) {
        tensorName[allTensors[i].get()] = net->tensorName()->GetAsString(i)->str();
    }
    // store buffer to STATIC model file
    genStaticModel(buffer, mnnFile, tensorName);
}
