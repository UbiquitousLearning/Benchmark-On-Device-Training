//
//  CPULayerNorm.cpp
//  MNN
//
//  Created by MNN on 2020/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>

#include "core/Execution.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUBackend.hpp"
#include "MNN_generated.h"


namespace MNN {

class CPULayerNorm : public Execution {
public:
    explicit CPULayerNorm(const MNN::Op* op, Backend* backend);
    virtual ~CPULayerNorm();

    ErrorCode onExecute(const std::vector<Tensor*> &inputs,  // NOLINT
                        const std::vector<Tensor*> &outputs) override;

    ErrorCode onResize(const std::vector<Tensor*> &inputs,  // NOLINT
                       const std::vector<Tensor*> &outputs) override;

private:
    std::vector<int> axis_;
    int inner_size_ = 1;
    int outter_size_ = 1;

    float epsilon_ = 0.001;

    std::unique_ptr<Tensor> gamma_;
    std::unique_ptr<Tensor> beta_;
};

CPULayerNorm::CPULayerNorm(const MNN::Op* op, Backend* backend)
        : Execution(backend) {
    const auto* layer_norm_param = op->main_as_LayerNorm();
    int axis_size = layer_norm_param->axis()->size();
    axis_.resize(axis_size);
    for (int i = 0; i < axis_size; ++i) {
        axis_[i] = layer_norm_param->axis()->Get(i);
    }

    epsilon_ = layer_norm_param->epsilon();

    int size = layer_norm_param->gamma()->size();
    gamma_.reset(Tensor::createDevice<float>({size}));
    auto status = backend->onAcquireBuffer(gamma_.get(), Backend::STATIC);
    if (!status) {
        MNN_ERROR("Out of memory when gamma is acquired in CPULayerNorm.\n");
    }
    const float* gamma_data = layer_norm_param->gamma()->data();
    memcpy(gamma_->host<float>(), gamma_data, size * sizeof(float));

    if (layer_norm_param->beta()->size() != size) {
        MNN_ERROR("Size of gamma and beta are not match in CPULayerNorm.\n");
    }
    beta_.reset(Tensor::createDevice<float>({size}));
    status = backend->onAcquireBuffer(beta_.get(), Backend::STATIC);
    if (!status) {
        MNN_ERROR("Out of memory when beta is acquired in CPULayerNorm.\n");
    }
    const float* beta_data = layer_norm_param->beta()->data();
    memcpy(beta_->host<float>(), beta_data, size * sizeof(float));
}

ErrorCode CPULayerNorm::onExecute(const std::vector<Tensor*> &inputs,
                                  const std::vector<Tensor*> &outputs) {
    const float* gamma = gamma_->host<float>();
    const float* beta = beta_->host<float>();

    const float* input = inputs.at(0)->host<float>();
    float* output = outputs.at(0)->host<float>();
    for (int i = 0; i < outter_size_; ++i) {
        const float* inner_input = input + i * inner_size_;
        float* inner_output = output + i * inner_size_;
        float sum = 0.f;
        for (int j = 0; j < inner_size_; ++j) {
            sum += inner_input[j];
        }
        float mean = sum / inner_size_;
        float square_sum = 0.f;
        for (int j = 0; j < inner_size_; ++j) {
            square_sum += (inner_input[j] - mean) * (inner_input[j] - mean);
        }
        float variable = square_sum / inner_size_;
        variable = 1.f / std::sqrt(variable + epsilon_);

        for (int j = 0; j < inner_size_; ++j) {
            inner_output[j] = (inner_input[j] - mean) * variable * gamma[j] + beta[j];
        }
    }
    return NO_ERROR;
}

ErrorCode CPULayerNorm::onResize(const std::vector<Tensor*> &inputs,
                                 const std::vector<Tensor*> &outputs) {
    outter_size_ = 1;
    inner_size_ = 1;
    int rank = inputs.at(0)->dimensions();
    std::vector<int> axis(axis_.size());
    for (int i = 0; i < axis_.size(); ++i) {
        if (axis_[i] < 0) {
            axis[i] += rank;
        }
    }
    std::sort(axis.begin(), axis.end());
    for (int i = 0; i < rank - axis.size(); ++i) {
        outter_size_ *= inputs.at(0)->length(i);
    }
    for (int i = rank - axis.size(); i < rank; ++i) {
        inner_size_ *= inputs.at(0)->length(i);
    }
    return NO_ERROR;
}

CPULayerNorm::~CPULayerNorm() {
    if (gamma_.get()) {
        backend()->onReleaseBuffer(gamma_.get(), Backend::STATIC);
    }
    if (beta_.get()) {
        backend()->onReleaseBuffer(beta_.get(), Backend::STATIC);
    }
}

class CPULayerNormCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs,
                        const MNN::Op* op, Backend* backend) const override {
        return new CPULayerNorm(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPULayerNormCreator, OpType_LayerNorm);

}  // namespace MNN
