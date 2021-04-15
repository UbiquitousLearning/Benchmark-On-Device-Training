//
//  quantizeWeight.hpp
//  MNN
//
//  Created by MNN on 2019/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef QUANTIZEWEIGHT_HPP
#define QUANTIZEWEIGHT_HPP
#include <stdint.h>
#include <vector>
#include <string>

// default: quantize weight every channel
int SymmetricQuantizeWeight(const float* weight, const int size, int8_t* quantizedWeight, float* scale,
                            const int channels, float weightClampValue);

// quantize convolution weight per channle
// firstly, multiply float weight by input_scale, then quantize the result to get input_sacle*weight_scale
// secondly, divide input_sacle*weight_scale by output_scale
int QuantizeConvPerChannel(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                           int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                           const std::vector<float>& outputScale, std::string method, float weightClampValue, bool mergeChannel = true);

int QuantizeDepthwiseConv(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                          int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                          const std::vector<float>& outputScale, std::string method, float weightClampValue);

#endif // QUANTIZEWEIGHT_HPP
