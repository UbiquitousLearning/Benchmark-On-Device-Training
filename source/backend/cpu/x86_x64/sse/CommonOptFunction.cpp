//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <emmintrin.h>
#include <string.h>
#include <algorithm>
#include "core/Macro.h"
#include "FunctionSummary.hpp"
void _SSE_MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count) {
    int countC16 = count / 16;
    int countR = count % 16;
    auto zero = _mm_set1_epi8(0);
    for (int i = 0; i < countC16; ++i) {
        auto s = _mm_castps_si128(_mm_loadu_ps((float*)source));
        auto d0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, s), 8);
        auto d1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, s), 8);
        _mm_storeu_ps((float*)dest, _mm_castsi128_ps(d0));
        _mm_storeu_ps((float*)dest + 4, _mm_castsi128_ps(d1));

        dest += 16;
        source += 16;
    }
    for (int i = 0; i < countR; ++i) {
        dest[i] = source[i];
    }
}

void _SSE_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    auto minV = _mm_set1_ps(6.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            dstV      = _mm_min_ps(dstV, minV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_storeu_ps(d, _mm_loadu_ps(s));
    }
}

void _SSE_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_storeu_ps(d, _mm_add_ps(_mm_loadu_ps(s), _mm_loadu_ps(d)));
    }
}

void _SSE_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    auto zero = _mm_set1_ps(0.0f);
    for (int j = 0; j < depthQuad; j++) {
        auto slopeZ       = _mm_loadu_ps(slope + 4 * j);
        const float* srcZ = src + 4 * j * sizeQuad;
        float* dstZ       = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            auto src   = _mm_loadu_ps(srcZ + 4 * i);
            auto mask0 = _mm_cmplt_ps(src, zero);
            auto mask1 = _mm_cmpge_ps(src, zero);
            auto other = _mm_mul_ps(src, slopeZ);
            _mm_storeu_ps(dstZ + 4 * i, _mm_add_ps(_mm_and_ps(other, mask0), _mm_and_ps(src, mask1)));
        }
    }
}

void _SSE_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep) {
    int dx, fx, fy;
    const int unit = 8;
    int widthUnit = width / unit;
    int widthRemain = width - widthUnit * unit;
    const float* weight_z = weight;
    bool need4 = widthRemain >= 4;
    if (need4) {
        widthRemain-=4;
    }
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < widthUnit; ++dx) {
            auto dstValue0 = _mm_set1_ps(0.0f);
            auto dstValue1 = _mm_set1_ps(0.0f);
            auto dstValue2 = _mm_set1_ps(0.0f);
            auto dstValue3 = _mm_set1_ps(0.0f);
            auto dstValue4 = _mm_set1_ps(0.0f);
            auto dstValue5 = _mm_set1_ps(0.0f);
            auto dstValue6 = _mm_set1_ps(0.0f);
            auto dstValue7 = _mm_set1_ps(0.0f);
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + 4 * fx;
                    auto weightValue = _mm_loadu_ps(weight_x);
                    dstValue0 = _mm_add_ps(dstValue0, _mm_mul_ps(_mm_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm_add_ps(dstValue1, _mm_mul_ps(_mm_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm_add_ps(dstValue2, _mm_mul_ps(_mm_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm_add_ps(dstValue3, _mm_mul_ps(_mm_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                    dstValue4 = _mm_add_ps(dstValue4, _mm_mul_ps(_mm_loadu_ps(src_x + 4 * src_w_setup), weightValue));
                    dstValue5 = _mm_add_ps(dstValue5, _mm_mul_ps(_mm_loadu_ps(src_x + 5 * src_w_setup), weightValue));
                    dstValue6 = _mm_add_ps(dstValue6, _mm_mul_ps(_mm_loadu_ps(src_x + 6 * src_w_setup), weightValue));
                    dstValue7 = _mm_add_ps(dstValue7, _mm_mul_ps(_mm_loadu_ps(src_x + 7 * src_w_setup), weightValue));
                }
            }
            _mm_storeu_ps(dstY + 4 * 0, dstValue0);
            _mm_storeu_ps(dstY + 4 * 1, dstValue1);
            _mm_storeu_ps(dstY + 4 * 2, dstValue2);
            _mm_storeu_ps(dstY + 4 * 3, dstValue3);
            _mm_storeu_ps(dstY + 4 * 4, dstValue4);
            _mm_storeu_ps(dstY + 4 * 5, dstValue5);
            _mm_storeu_ps(dstY + 4 * 6, dstValue6);
            _mm_storeu_ps(dstY + 4 * 7, dstValue7);
            dstY += 4 * unit;
            srcY += unit * src_w_setup;
        }
        if (need4) {
            auto dstValue0 = _mm_set1_ps(0.0f);
            auto dstValue1 = _mm_set1_ps(0.0f);
            auto dstValue2 = _mm_set1_ps(0.0f);
            auto dstValue3 = _mm_set1_ps(0.0f);
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + 4 * fx;
                    auto weightValue = _mm_loadu_ps(weight_x);
                    dstValue0 = _mm_add_ps(dstValue0, _mm_mul_ps(_mm_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm_add_ps(dstValue1, _mm_mul_ps(_mm_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm_add_ps(dstValue2, _mm_mul_ps(_mm_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm_add_ps(dstValue3, _mm_mul_ps(_mm_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                }
            }
            _mm_storeu_ps(dstY + 4 * 0, dstValue0);
            _mm_storeu_ps(dstY + 4 * 1, dstValue1);
            _mm_storeu_ps(dstY + 4 * 2, dstValue2);
            _mm_storeu_ps(dstY + 4 * 3, dstValue3);
            dstY += 4 * 4;
            srcY += 4 * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * 4;
            auto dstValue = _mm_set1_ps(0.0f);
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 4 * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm_add_ps(dstValue, _mm_mul_ps(_mm_loadu_ps(src_x), _mm_loadu_ps(weight_x)));
                }
            }
            _mm_storeu_ps(dst_x, dstValue);
        }
    }
}

void _SSE_MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    auto count = countC8 * 2;
    auto p0    = _mm_set1_ps(parameters[0]);
    auto p1    = _mm_set1_ps(parameters[1]);
    auto p2    = _mm_set1_ps(parameters[2]);
    auto p3    = _mm_set1_ps(parameters[3]);
    auto p4    = _mm_set1_ps(parameters[4]);
    auto p5    = _mm_set1_ps(parameters[5]);
    auto p6    = _mm_set1_ps(parameters[6]);
    auto p7    = _mm_set1_ps(parameters[7]);
    auto xMax  = _mm_set1_ps(87);
    auto xMin  = _mm_set1_ps(-87);
    auto basic = _mm_set1_epi32(1 << 23);
    for (int i = 0; i < count; ++i) {
        auto x            = _mm_xor_ps(_mm_loadu_ps(source + i * 4), _mm_set1_ps(-0.f));
        x                 = _mm_max_ps(x, xMin);
        x                 = _mm_min_ps(x, xMax);
        auto div          = _mm_mul_ps(x, p1);
        auto divInt       = _mm_cvtps_epi32(div);
        div               = _mm_cvtepi32_ps(divInt);
        auto div2         = _mm_add_epi32(divInt, _mm_set1_epi32(127));
        div2 = _mm_mullo_epi32(div2, basic);
        auto expBasic  = _mm_castsi128_ps(div2);
        auto xReamin   = _mm_sub_ps(x, _mm_mul_ps(div, p0));
        auto t         = xReamin;
        auto c0        = _mm_mul_ps(p7, t);
        auto c1        = _mm_add_ps(c0, p6);
        auto c2        = _mm_mul_ps(c1, t);
        auto c3        = _mm_add_ps(c2, p5);
        auto c4        = _mm_mul_ps(c3, t);
        auto c5        = _mm_add_ps(c4, p4);
        auto c6        = _mm_mul_ps(c5, t);
        auto c7        = _mm_add_ps(c6, p3);
        auto c8        = _mm_mul_ps(c7, t);
        auto c9        = _mm_add_ps(c8, p2);
        auto expRemain = c9;
        _mm_store_ps(dest + 4 * i, _mm_mul_ps(expBasic, expRemain));
    }
}

void _SSE_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    __m128i zero = _mm_set1_epi32(0);
    __m128 minValue = _mm_set1_ps(minV);
    __m128 maxValue = _mm_set1_ps(maxV);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    __m128 scaleValue = _mm_loadu_ps(scalep);
    int32_t temp[4];

    for (int i = 0; i < sizeQuad; ++i) {
        __m128 f0 = _mm_loadu_ps(src + 4 * i);
        f0 = _mm_mul_ps(f0, scaleValue);
        f0 = _mm_min_ps(f0, maxValue);
        f0 = _mm_max_ps(f0, minValue);
        auto m0 = _mm_cmplt_ps(f0, _mm_castsi128_ps(zero));
        m0 = _mm_blendv_ps(plus, minus, m0);
        f0 = _mm_add_ps(f0, m0);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
        *(__m128i*)temp = d0;
        for (int j=0; j<4; ++j) {
            dst[4*i+j] = temp[j];
        }
    }
}

void _SSE_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 4;
    auto sizeRemain = sizeQuad % 4;
    __m128i zero = _mm_set1_epi32(0);
    __m128 scaleValue = _mm_loadu_ps(scale);
    for (int i = 0; i < sizeC4; ++i) {
        auto s = _mm_castps_si128(_mm_loadu_ps((const float*)(src)));
        auto s0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, s), 8);
        auto s1_16 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, s), 8);
        auto s0_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s0_16), 16);
        auto s1_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s0_16), 16);
        auto s2_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s1_16), 16);
        auto s3_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s1_16), 16);
        auto s0_f = _mm_cvtepi32_ps(s0_32);
        auto s1_f = _mm_cvtepi32_ps(s1_32);
        auto s2_f = _mm_cvtepi32_ps(s2_32);
        auto s3_f = _mm_cvtepi32_ps(s3_32);
        _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 2, _mm_mul_ps(s2_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 3, _mm_mul_ps(s3_f, scaleValue));
        src += 16;
        dst += 16;
    }
    if (sizeRemain > 0) {
        int8_t srcTemp[128];
        ::memcpy(srcTemp, src, sizeRemain * 4);
        auto s = *(__m128i*)srcTemp;
        auto s0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, s), 8);
        auto s1_16 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, s), 8);
        auto s0_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s0_16), 16);
        auto s1_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s0_16), 16);
        auto s2_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s1_16), 16);
        auto s3_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s1_16), 16);
        auto s0_f = _mm_cvtepi32_ps(s0_32);
        auto s1_f = _mm_cvtepi32_ps(s1_32);
        auto s2_f = _mm_cvtepi32_ps(s2_32);
        auto s3_f = _mm_cvtepi32_ps(s3_32);
        switch (sizeRemain) {
            case 3:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 2, _mm_mul_ps(s2_f, scaleValue));
                break;
            case 2:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
                break;
            case 1:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                break;
            default:
                break;
        }
    }
}

void _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    int widthC4 = width / 4;
    int widthRemain = width % 4;
    auto weight = (const int16_t*)weightO;
    auto biasValue = _mm_castps_si128(_mm_loadu_ps((const float*)parameters->bias));
    //auto biasValue = *(__m128i*)parameters->bias;
    auto scaleValue = _mm_loadu_ps((const float*)parameters->scale);
    __m128i d0, d1, d2, d3;
    int dx, fx, fy;
    __m128i srcValue0;
    auto srcTemp0 = (int64_t*)(&srcValue0);
    __m128i srcValue1;
    auto srcTemp1 = (int64_t*)(&srcValue1);
    __m128i weightValue;
    auto weightTemp = (int64_t*)(&weightValue);
    __m128i zero = _mm_xor_si128(srcValue1, srcValue1);
    __m128 zero128 = _mm_set1_ps(0.0f);
    auto minValue = _mm_set1_epi8(parameters->minValue);
    auto maxValue = _mm_set1_epi8(parameters->maxValue);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    if (4 == src_w_step) {
        // Stride = 1
        for (dx = 0; dx < widthC4; ++dx) {
            d0 = biasValue;
            d1 = biasValue;
            d2 = biasValue;
            d3 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    auto s0_16 = _mm_castps_si128(_mm_loadu_ps((float*)src_x));
                    auto s1_16 = _mm_castps_si128(_mm_loadu_ps((float*)src_x + 4));
                    auto s0_32 = _mm_unpacklo_epi16(s0_16, zero);
                    auto s1_32 = _mm_unpackhi_epi16(s0_16, zero);
                    auto s2_32 = _mm_unpacklo_epi16(s1_16, zero);
                    auto s3_32 = _mm_unpackhi_epi16(s1_16, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                    d2 = _mm_add_epi32(d2, _mm_madd_epi16(weightValue, s2_32));
                    d3 = _mm_add_epi32(d3, _mm_madd_epi16(weightValue, s3_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(d0));
            dst += 16;
            src += src_w_step * 4;
        }
    } else {
        for (dx = 0; dx < widthC4; ++dx) {
            d0 = biasValue;
            d1 = biasValue;
            d2 = biasValue;
            d3 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + src_w_step * 1);
                    srcTemp1[0] = *(int64_t*)(src_x + src_w_step * 2);
                    srcTemp1[1] = *(int64_t*)(src_x + src_w_step * 3);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm_unpackhi_epi16(srcValue0, zero);
                    auto s2_32 = _mm_unpacklo_epi16(srcValue1, zero);
                    auto s3_32 = _mm_unpackhi_epi16(srcValue1, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                    d2 = _mm_add_epi32(d2, _mm_madd_epi16(weightValue, s2_32));
                    d3 = _mm_add_epi32(d3, _mm_madd_epi16(weightValue, s3_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(d0));
            dst += 16;
            src += src_w_step * 4;
        }
    }
    switch (widthRemain) {
        case 3:
        {
            d0 = biasValue;
            d1 = biasValue;
            d2 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + src_w_step * 1);
                    srcTemp1[0] = *(int64_t*)(src_x + src_w_step * 2);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm_unpackhi_epi16(srcValue0, zero);
                    auto s2_32 = _mm_unpacklo_epi16(srcValue1, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                    d2 = _mm_add_epi32(d2, _mm_madd_epi16(weightValue, s2_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int8_t temp[128];
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(d0));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        case 2:
        {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + src_w_step * 1);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm_unpackhi_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int8_t temp[128];
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(d0));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        case 1:
        {
            d0 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            f0 = _mm_mul_ps(f0, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            f0 = _mm_add_ps(f0, m0);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d0 = _mm_packs_epi16(d0, d2);
            int8_t temp[128];
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(d0));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        default:
            break;
    }
}
void _SSE_MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            auto sumValue = _mm_set1_ps(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = _mm_add_ps(sumValue, _mm_mul_ps(_mm_loadu_ps(A + x * 4), _mm_loadu_ps(by + x * 4)));
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            sumValue = _mm_hadd_ps(sumValue, sumValue);
            sumValue = _mm_hadd_ps(sumValue, sumValue);
            auto s = _mm_cvtss_f32(sumValue);
            C[y] = sumRemain + s;
        }
    } else {
        auto hC4 = h / 4;
        auto hR = hC4 * 4;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 4 * y;
            auto sumValue = _mm_set1_ps(0.0f);
            if (biasPtr != nullptr) {
                sumValue = _mm_loadu_ps(biasPtr + 4 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = _mm_add_ps(sumValue, _mm_mul_ps(_mm_set1_ps(A[x]), _mm_loadu_ps(bs + h * x)));
            }
            _mm_storeu_ps(C + 4 * y, sumValue);
        }
        for (int y=hR + tId; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}

extern "C" {
void MNNInt8ToUInt8(void* ptr, int count) {
    auto src = (int8_t*)ptr;
    auto dst = (uint8_t*)ptr;
    int c16 = count / 16;
    count = count % 16;
    auto zero = _mm_set1_epi8(0);
    auto offset = _mm_set1_epi16(128);
    for (int v = 0; v < c16; ++v) {
        auto i8Value = _mm_loadu_si128((__m128i*)(src));
        auto i16Value0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, i8Value), 8);
        auto i16Value1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, i8Value), 8);
        i16Value0 = _mm_add_epi16(i16Value0, offset);
        i16Value1 = _mm_add_epi16(i16Value1, offset);
        i8Value = _mm_packus_epi16(i16Value0, i16Value1);
        _mm_storeu_si128((__m128i*)dst, i8Value);
        dst += 16;
        src += 16;
    }
    for (int v = 0; v < count; ++v) {
        dst[v] = (int)src[v] + 128;
    }
}
}
