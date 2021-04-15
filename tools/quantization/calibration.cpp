//
//  calibration.cpp
//  MNN
//
//  Created by MNN on 2019/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "calibration.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <algorithm>
#include <MNN/ImageProcess.hpp>
#include "flatbuffers/util.h"
#include "logkit.h"
#include "quantizeWeight.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "Helper.hpp"
#include "core/TensorUtils.hpp"

using namespace MNN::CV;

Calibration::Calibration(MNN::NetT* model, const uint8_t* modelBuffer, const int bufferSize, const std::string& configPath)
    : _originaleModel(model) {
    // when the format of input image is RGB/BGR, channels equal to 3, GRAY is 1
    int channles = 3;

    rapidjson::Document document;
    {
        std::ifstream fileNames(configPath.c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return;
        }
    }
    auto picObj = document.GetObject();
    ImageProcess::Config config;
    config.filterType = BILINEAR;
    config.destFormat = BGR;
    {
        if (picObj.HasMember("format")) {
            auto format = picObj["format"].GetString();
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}, {"RGBA", RGBA}, {"BGRA", BGRA}};
            if (formatMap.find(format) != formatMap.end()) {
                config.destFormat = formatMap.find(format)->second;
            }
        }
    }

    switch (config.destFormat) {
        case GRAY:
            channles = 1;
            break;
        case RGB:
        case BGR:
            channles = 3;
            break;
        case RGBA:
        case BGRA:
            channles = 4;
            break;
        default:
            break;
    }

    config.sourceFormat = RGBA;
    std::string imagePath;
    _imageNum = 0;
    {
        if (picObj.HasMember("mean")) {
            auto mean = picObj["mean"].GetArray();
            int cur   = 0;
            for (auto iter = mean.begin(); iter != mean.end(); iter++) {
                config.mean[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("normal")) {
            auto normal = picObj["normal"].GetArray();
            int cur     = 0;
            for (auto iter = normal.begin(); iter != normal.end(); iter++) {
                config.normal[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("width")) {
            _width = picObj["width"].GetInt();
        }
        if (picObj.HasMember("height")) {
            _height = picObj["height"].GetInt();
        }
        if (picObj.HasMember("path")) {
            imagePath = picObj["path"].GetString();
        }
        if (picObj.HasMember("used_image_num")) {
            _imageNum = picObj["used_image_num"].GetInt();
        }
        if (picObj.HasMember("feature_quantize_method")) {
            std::string method = picObj["feature_quantize_method"].GetString();
            if (Helper::featureQuantizeMethod.find(method) != Helper::featureQuantizeMethod.end()) {
                _featureQuantizeMethod = method;
            } else {
                MNN_ERROR("not supported feature quantization method: %s\n", method.c_str());
                return;
            }
        }
        if (picObj.HasMember("weight_quantize_method")) {
            std::string method = picObj["weight_quantize_method"].GetString();
            if (Helper::weightQuantizeMethod.find(method) != Helper::weightQuantizeMethod.end()) {
                _weightQuantizeMethod = method;
            } else {
                MNN_ERROR("not supported weight quantization method: %s\n", method.c_str());
                return;
            }
        }
        DLOG(INFO) << "Use feature quantization method: " << _featureQuantizeMethod;
        DLOG(INFO) << "Use weight quantization method: " << _weightQuantizeMethod;
        if (picObj.HasMember("feature_clamp_value")) {
            float value = (int)picObj["feature_clamp_value"].GetFloat();
            if (value < 0.0f || value > 127.0f) {
                MNN_ERROR("feature_clamp_value should be in (0, 127], got: %f\n", value);
                return;
            }
            _featureClampValue = value;
        }
        if (picObj.HasMember("weight_clamp_value")) {
            float value = (int)picObj["weight_clamp_value"].GetFloat();
            if (value < 0.0f || value > 127.0f) {
                MNN_ERROR("weight_clamp_value should be in (0, 127], got: %f\n", value);
                return;
            }
            _weightClampValue = value;
        }
        DLOG(INFO) << "feature_clamp_value: " << _featureClampValue;
        DLOG(INFO) << "weight_clamp_value: " << _weightClampValue;
        if (picObj.HasMember("skip_quant_op_names")) {
            auto skip_quant_op_names = picObj["skip_quant_op_names"].GetArray();
            for (auto iter = skip_quant_op_names.begin(); iter != skip_quant_op_names.end(); iter++) {
                std::string skip_quant_op_name = iter->GetString();
                _skip_quant_ops.emplace_back(skip_quant_op_name);
                DLOG(INFO) << "skip quant op name: " << skip_quant_op_name;
            }
        }
        if (picObj.HasMember("debug")) {
            _debug = picObj["debug"].GetBool();
        }
    }
    std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
    _process = process;

    // read images file names
    Helper::readImages(_imgaes, imagePath.c_str(), &_imageNum);

    _initMNNSession(modelBuffer, bufferSize, channles);
    _initMaps();
}

void Calibration::_initMNNSession(const uint8_t* modelBuffer, const int bufferSize, const int channels) {
    _interpreterOrigin.reset(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    MNN::ScheduleConfig config;
    _sessionOrigin     = _interpreterOrigin->createSession(config);
    _inputTensorOrigin = _interpreterOrigin->getSessionInput(_sessionOrigin, NULL);

    _fake_quant_weights();

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, _originaleModel);
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto buffer = builder.GetBufferPointer();

    _interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    _session     = _interpreter->createSession(config);
    _inputTensor = _interpreter->getSessionInput(_session, NULL);

    _inputTensorDims.resize(4);
    auto inputTensorDataFormat = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
    if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        _inputTensorDims[0] = 1;
        _inputTensorDims[1] = _height;
        _inputTensorDims[2] = _width;
        _inputTensorDims[3] = channels;
    } else {
        _inputTensorDims[0] = 1;
        _inputTensorDims[1] = channels;
        _inputTensorDims[2] = _height;
        _inputTensorDims[3] = _width;
    }
    if (_featureQuantizeMethod == "KL") {
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);
    } else if (_featureQuantizeMethod == "ADMM") {
        DCHECK((_imageNum * 4 * _height * _width) < (INT_MAX / 4)) << "Use Little Number of Images When Use ADMM";
        _inputTensorDims[0] = _imageNum;
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);
    }
}

void Calibration::_initMaps() {
    _featureInfo.clear();
    _featureInfoOrigin.clear();
    _opInfo.clear();
    _tensorMap.clear();
    // run mnn once, initialize featureMap, opInfo map
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        _opInfo[opName].first = nTensors;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo after = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        _opInfo[opName].second = nTensors;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return true;
    };
    _interpreter->runSessionWithCallBackInfo(_session, before, after);


    MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo afterOrigin = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return true;
    };
    _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

    for (auto& op : _originaleModel->oplists) {
        if (_opInfo.find(op->name) == _opInfo.end()) {
            continue;
        }
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            _tensorMap[op->inputIndexes[i]] = _opInfo[op->name].first[i];
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            _tensorMap[op->outputIndexes[i]] = _opInfo[op->name].second[i];
        }
    }

    if (_featureQuantizeMethod == "KL") {
        // set the tensor-statistic method of input tensor as THRESHOLD_MAX
        auto inputTensorStatistic = _featureInfo.find(_inputTensor);
        if (inputTensorStatistic != _featureInfo.end()) {
            inputTensorStatistic->second->setThresholdMethod(THRESHOLD_MAX);
        }
    }
}

void Calibration::_computeFeatureMapsRange() {
    // feed input data according to input images
    int count = 0;
    for (const auto& img : _imgaes) {
        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        count++;
        Helper::preprocessInput(_process.get(), _width, _height, img, _inputTensor);

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };

        _interpreter->runSessionWithCallBackInfo(_session, before, after);
        MNN_PRINT("\rComputeFeatureRange: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_collectFeatureMapsDistribution() {
    for (auto& iter : _featureInfo) {
        iter.second->resetDistribution();
    }
    // feed input data according to input images
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->updateDistribution();
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->updateDistribution();
                }
            }
        }
        return true;
    };
    int count = 0;
    for (const auto& img : _imgaes) {
        count++;

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedDistributionFlag();
        }
        Helper::preprocessInput(_process.get(), _width, _height, img, _inputTensor);
        _interpreter->runSessionWithCallBackInfo(_session, before, after);

        MNN_PRINT("\rCollectFeatureDistribution: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_computeFeatureScaleKL() {
    _computeFeatureMapsRange();
    _collectFeatureMapsDistribution();

    _scales.clear();
    for (auto& iter : _featureInfo) {
        AUTOTIME;
        _scales[iter.first] = iter.second->finishAndCompute();
    }
    //_featureInfo.clear();//No need now
}

void Calibration::_computeFeatureScaleADMM() {
    // feed input data according to input images
    int count                           = 0;
    std::vector<int> oneImageTensorDims = _inputTensorDims;
    oneImageTensorDims[0]               = 1;
    auto inputTensorDataFormat          = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
    auto dimType                        = MNN::Tensor::CAFFE_C4;
    if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        dimType = MNN::Tensor::TENSORFLOW;
    }

    for (const auto& img : _imgaes) {
        auto curPtr = _inputTensor->host<float>() + count * _inputTensor->stride(0);
        std::shared_ptr<MNN::Tensor> tensorWarp(
            MNN::Tensor::create(oneImageTensorDims, _inputTensor->getType(), curPtr, dimType));
        Helper::preprocessInput(_process.get(), _width, _height, img, tensorWarp.get());

        count++;
        MNN_PRINT("\rProcessImage: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
    _scales.clear();

    const int totalLayers = _featureInfo.size();
    count                 = 0;

    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };

    _interpreter->runSessionWithCallBackInfo(_session, before, after);
    MNN_PRINT("\n");
}

void Calibration::_updateScale() {
    for (const auto& op : _originaleModel->oplists) {
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
            opType != MNN::OpType_Eltwise) {
            continue;
        }
        auto tensorsPair = _opInfo.find(op->name);
        if (tensorsPair == _opInfo.end()) {
            MNN_ERROR("Can't find tensors for %s\n", op->name.c_str());
        }

        if (opType == MNN::OpType_Eltwise) {
            auto param = op->main.AsEltwise();
            // Now only support AddInt8
            if (param->type != MNN::EltwiseType_SUM) {
                continue;
            }
            const auto& inputScale0   = _scales[tensorsPair->second.first[0]];
            const auto& inputScale1   = _scales[tensorsPair->second.first[1]];
            const auto& outputScale   = _scales[tensorsPair->second.second[0]];
            const int outputScaleSize = outputScale.size();
            std::vector<float> outputInvertScale(outputScaleSize);
            Helper::invertData(outputInvertScale.data(), outputScale.data(), outputScaleSize);
            op->type = MNN::OpType_EltwiseInt8;
            op->main.Reset();
            op->main.type = MNN::OpParameter_EltwiseInt8;

            auto eltwiseInt8Param         = new MNN::EltwiseInt8T;
            auto input0ScaleParam         = new MNN::QuantizedFloatParamT;
            auto input1ScaleParam         = new MNN::QuantizedFloatParamT;
            auto outputScaleParam         = new MNN::QuantizedFloatParamT;
            input0ScaleParam->tensorScale = inputScale0;
            input1ScaleParam->tensorScale = inputScale1;
            outputScaleParam->tensorScale = outputInvertScale;
            eltwiseInt8Param->inputQuan0  = std::unique_ptr<MNN::QuantizedFloatParamT>(input0ScaleParam);
            eltwiseInt8Param->inputQuan1  = std::unique_ptr<MNN::QuantizedFloatParamT>(input1ScaleParam);
            eltwiseInt8Param->outputQuan  = std::unique_ptr<MNN::QuantizedFloatParamT>(outputScaleParam);
            op->main.value                = eltwiseInt8Param;

            continue;
        }

        // below is Conv/DepthwiseConv
        const auto& inputScale  = _scales[tensorsPair->second.first[0]];
        const auto& outputScale = _scales[tensorsPair->second.second[0]];

        auto param                = op->main.AsConvolution2D();
        const int channles        = param->common->outputCount;
        const int weightSize      = param->weight.size();
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        auto& quantizedParam = param->symmetricQuan;
        quantizedParam->scale.resize(channles);
        quantizedParam->weight.resize(weightSize);
        quantizedParam->bias.resize(channles);

        if (opType == MNN::OpType_Convolution) {
            QuantizeConvPerChannel(param->weight.data(), param->weight.size(), param->bias.data(),
                                   quantizedParam->weight.data(), quantizedParam->bias.data(),
                                   quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod, _weightClampValue);
            op->type = MNN::OpType_ConvInt8;

        } else if (opType == MNN::OpType_ConvolutionDepthwise) {
            QuantizeDepthwiseConv(param->weight.data(), param->weight.size(), param->bias.data(),
                                  quantizedParam->weight.data(), quantizedParam->bias.data(),
                                  quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod, _weightClampValue);
            op->type = MNN::OpType_DepthwiseConvInt8;
        }
        if (param->common->relu6) {
            param->common->relu  = true;
            param->common->relu6 = false;
        }
        param->weight.clear();
        param->bias.clear();
    }
}

void Calibration::_insertDequantize() {
    // Search All Int Tensors
    std::set<int> int8Tensors;
    std::set<int> int8Outputs;
    for (auto& op : _originaleModel->oplists) {
        if (Helper::INT8SUPPORTED_OPS.count(op->type) > 0) {
            for (auto index : op->inputIndexes) {
                int8Tensors.insert(index);
            }
            for (auto index : op->outputIndexes) {
                int8Tensors.insert(index);
                int8Outputs.insert(index);
            }
        }
    }
    for (auto& op : _originaleModel->oplists) {
        for (auto index : op->inputIndexes) {
            auto iter = int8Outputs.find(index);
            if (iter != int8Outputs.end()) {
                int8Outputs.erase(iter);
            }
        }
    }

    // Insert Convert For Not Support Int8 Ops
    for (auto iter = _originaleModel->oplists.begin(); iter != _originaleModel->oplists.end();) {
        auto op           = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;
        // check whether is output op
        // if Yes, insert dequantization op after this op
        if (Helper::INT8SUPPORTED_OPS.find(opType) != Helper::INT8SUPPORTED_OPS.end()) {
            // this is quantized op
            iter++;
            continue;
        }

        auto& inputIndexes  = op->inputIndexes;
        const int inputSize = inputIndexes.size();

        // insert dequantization op before this op
        for (int i = 0; i < inputSize; ++i) {
            const auto curInputIndex = inputIndexes[i];
            if (int8Tensors.find(curInputIndex) == int8Tensors.end()) {
                continue;
            }
            auto input        = _tensorMap[curInputIndex];
            auto inputOpScale = _scales[input];

            // construct new op
            auto dequantizationOp       = new MNN::OpT;
            dequantizationOp->main.type = MNN::OpParameter_QuantizedFloatParam;
            dequantizationOp->name      = "___Int8ToFloat___For_" + name + flatbuffers::NumToString(i);

            dequantizationOp->type           = MNN::OpType_Int8ToFloat;
            auto dequantizationParam         = new MNN::QuantizedFloatParamT;
            dequantizationOp->main.value     = dequantizationParam;
            dequantizationParam->tensorScale = inputOpScale;

            dequantizationOp->inputIndexes.push_back(curInputIndex);
            dequantizationOp->outputIndexes.push_back(_originaleModel->tensorName.size());
            _originaleModel->tensorName.push_back(dequantizationOp->name);

            // reset current op's input index at i
            inputIndexes[i] = dequantizationOp->outputIndexes[0];

            iter = _originaleModel->oplists.insert(iter, std::unique_ptr<MNN::OpT>(dequantizationOp));
            iter++;
        }

        iter++;
        // LOG(INFO) << "insert quantization op after this op if neccessary";
        // insert quantization op after this op if neccessary
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            const auto outputIndex = op->outputIndexes[i];
            if (int8Tensors.find(outputIndex) == int8Tensors.end()) {
                continue;
            }
            auto output   = _tensorMap[outputIndex];
            auto curScale = _scales[output];
            // construct one quantization op(FloatToInt8)
            auto quantizationOp        = new MNN::OpT;
            quantizationOp->main.type  = MNN::OpParameter_QuantizedFloatParam;
            quantizationOp->name       = name + "___FloatToInt8___" + flatbuffers::NumToString(i);
            quantizationOp->type       = MNN::OpType_FloatToInt8;
            auto quantizationParam     = new MNN::QuantizedFloatParamT;
            quantizationOp->main.value = quantizationParam;

            const int channels = curScale.size();
            std::vector<float> quantizationScale(channels);
            Helper::invertData(quantizationScale.data(), curScale.data(), channels);
            quantizationParam->tensorScale = quantizationScale;

            quantizationOp->inputIndexes.push_back(_originaleModel->tensorName.size());
            quantizationOp->outputIndexes.push_back(outputIndex);
            _originaleModel->tensorName.push_back(_originaleModel->tensorName[outputIndex]);
            _originaleModel->tensorName[outputIndex] = quantizationOp->name;
            op->outputIndexes[i]                              = quantizationOp->inputIndexes[0];

            iter = _originaleModel->oplists.insert(iter, std::unique_ptr<MNN::OpT>(quantizationOp));
            iter++;
        }
    }

    // Insert Turn float Op for output
    for (auto index : int8Outputs) {
        // construct new op
        auto dequantizationOp       = new MNN::OpT;
        dequantizationOp->main.type = MNN::OpParameter_QuantizedFloatParam;
        dequantizationOp->name      = "___Int8ToFloat___For_" + flatbuffers::NumToString(index);

        dequantizationOp->type           = MNN::OpType_Int8ToFloat;
        auto dequantizationParam         = new MNN::QuantizedFloatParamT;
        dequantizationOp->main.value     = dequantizationParam;
        dequantizationParam->tensorScale = _scales[_tensorMap[index]];

        dequantizationOp->inputIndexes.push_back(index);
        dequantizationOp->outputIndexes.push_back(_originaleModel->tensorName.size());
        auto originTensorName              = _originaleModel->tensorName[index];
        _originaleModel->tensorName[index] = dequantizationOp->name;
        _originaleModel->tensorName.emplace_back(originTensorName);

        _originaleModel->oplists.insert(_originaleModel->oplists.end(), std::unique_ptr<MNN::OpT>(dequantizationOp));
    }
}

void Calibration::_fake_quant_weights() {
    auto findAbsMax = [&] (const float* weights, const int size) {
        float absMax = 0;
        for (int i = 0; i < size; i++) {
            if (std::fabs(weights[i]) > absMax) {
                absMax = std::fabs(weights[i]);
            }
        }

        return absMax;
    };

    for (const auto& op : _originaleModel->oplists) {
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }

        auto param = op->main.AsConvolution2D();
        const int kernelNum = param->common->outputCount;
        std::vector<float> weights = param->weight;
        const int weightSize = weights.size();
        const int kernelSize = weightSize / kernelNum;

        for (int i = 0; i < kernelNum; i++) {
            const int offset = i * kernelSize;
            float absMax = findAbsMax(weights.data() + offset, kernelSize);
            float scale = absMax / _weightClampValue;
            if (absMax < 1e-6f) {
                scale = absMax;
            }

            for (int j = 0; j < kernelSize; j++) {
                float value = weights[offset + j];
                float quantValue = std::round(value / scale);
                float clampedValue = std::max(std::min(quantValue, _weightClampValue), -_weightClampValue);
                float dequantValue = scale * clampedValue;
                param->weight[offset + j] = dequantValue;
            }
        }
    }
}

void Calibration::_computeQuantError() {
    int count = 0;
    std::map<std::string, std::vector<float>> overflowRatiosMap;
    std::map<std::string, std::vector<float>> tensorCosDistanceMap;

    std::vector<int> inputShape = {1, _inputTensorDims[1], _inputTensorDims[2], _inputTensorDims[3]};
    _interpreter->resizeTensor(_inputTensor, inputShape);
    _interpreter->resizeSession(_session);
    _interpreterOrigin->resizeTensor(_inputTensorOrigin, inputShape);
    _interpreterOrigin->resizeSession(_sessionOrigin);

    for (const auto& img : _imgaes) {
        count++;
        Helper::preprocessInput(_process.get(), _width, _height, img, _inputTensor);

        std::map<std::string, std::vector<float>> fakeQuantedFeatures;

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                        fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                        fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                    }
                }
            }
            return true;
        };

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        _interpreter->runSessionWithCallBackInfo(_session, before, after);

        Helper::preprocessInput(_process.get(), _width, _height, img, _inputTensorOrigin);

        MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto name = _featureInfoOrigin[t]->name();
                        float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                        tensorCosDistanceMap[name].emplace_back(cosDis);
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo afterOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto name = _featureInfoOrigin[t]->name();
                        float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                        tensorCosDistanceMap[name].emplace_back(cosDis);
                    }
                }
            }
            return true;
        };

        for (auto& iter : _featureInfoOrigin) {
            iter.second->setVisited(false);
        }

        _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

        MNN_PRINT("\rcomputeDistance: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n\nDebug info:\n\n");

    for (auto& iter : tensorCosDistanceMap) {
        auto name = iter.first;
        float sumCos = 0.0f, sumOverflow = 0.0f;
        for (int i = 0; i < iter.second.size(); i++) {
            sumCos += iter.second[i];
            sumOverflow += overflowRatiosMap[name][i];
        }
        float avgCosDistance = sumCos / _imgaes.size();
        float avgOverflowRatio = sumOverflow / _imgaes.size();

        MNN_PRINT("%s:  cos distance: %f, overflow ratio: %f\n", name.c_str(), avgCosDistance, avgOverflowRatio);
    }
}

void Calibration::runQuantizeModel() {
    if (_featureQuantizeMethod == "KL") {
        _computeFeatureScaleKL();
    } else if (_featureQuantizeMethod == "ADMM") {
        _computeFeatureScaleADMM();
    }
    if (_debug) {
        _computeQuantError();
    }
    _updateScale();
    _insertDequantize();
}

void Calibration::dumpTensorScales(const std::string& modelFile) {
    rapidjson::StringBuffer sb;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);

    writer.StartArray();

    for (auto iter = _originaleModel->oplists.begin(); iter != _originaleModel->oplists.end(); iter++) {
        auto op           = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;
        
        if (opType == MNN::OpType_Raster) {
            continue;
        }

        writer.StartObject();

        writer.Key("name");
        writer.String(rapidjson::StringRef(name.c_str(), name.size()));

        auto& inputIndexes  = op->inputIndexes;
        const int inputSize = inputIndexes.size();

        if (inputSize > 0) {
            writer.Key("inputs");
            writer.StartArray();
            for (int i = 0; i < inputSize; ++i) {
                const auto curInputIndex = inputIndexes[i];
                
                auto input        = _tensorMap[curInputIndex];
                auto inputOpScale = _scales[input];
                
                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curInputIndex);

                writer.Key("scales");
                writer.StartArray();
                for(auto scale : inputOpScale) {
                    writer.Double(scale);
                }
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }
 
        auto& outputIndexes  = op->outputIndexes;
        const int outputSize = outputIndexes.size();

        if (outputSize > 0) {
            writer.Key("outputs");
            writer.StartArray();
            for (int i = 0; i < outputSize; ++i) {
                const auto curOutputIndex = outputIndexes[i];
                
                auto output        = _tensorMap[curOutputIndex];
                auto outputOpScale = _scales[output];
                
                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curOutputIndex);

                writer.Key("scales");
                writer.StartArray();
                for(auto scale : outputOpScale) {
                    writer.Double(scale);
                }
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }

        writer.EndObject();
    }
    writer.EndArray();

    std::string scaleFile = modelFile + ".json";
    std::ofstream os(scaleFile);
    if (os.is_open()) {
        os << sb.GetString() << std::endl;
        os.close();
    } else {
        std::cerr << "open scale file " << scaleFile << " fail. error code:" << os.failbit << std::endl;
    }
}
