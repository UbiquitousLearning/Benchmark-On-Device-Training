//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonOptFunction.h"
#include <string.h>
#include <algorithm>
#include <math.h>
#include "math/Vec.hpp"
#include <vector>
int MNNGetC4DivNumber(int h) {
    auto remain = h % 4;
    if (0 == remain) {
        return h / 4;
    }
    if (4 % remain == 0) {
        return h / remain;
    }
    return h;
}

#ifndef MNN_USE_SSE
#ifndef MNN_USE_NEON
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}
#endif
bool MNNReorder4x4ByPlatform(float* dst, size_t size) {
    // Do nothing
    return false;
}
void MNNFunctionInit() {
    // Do nothing
}
#endif

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define UNIT 4
using Vec4 = MNN::Math::Vec<float, 4>;

void MNNScaleAndAddBiasOutside(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                               size_t biasNumber) {
    for (size_t p = 0; p < planeNumber; ++p) {
        float* dstPlane       = dst + p * biasNumber;
        const float* srcPlane = src + p * biasNumber;
        for (int z = 0; z < biasNumber; ++z) {
            dstPlane[z] = srcPlane[z] * alpha[z] + bias[z];
        }
    }
}



#ifndef MNN_USE_NEON

#ifndef MNN_USE_SSE
void MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ        = dst + planeNumber * 4 * z;
        const float* biasZ = bias + 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX = dstZ + 4 * p;
            for (int i = 0; i < 4; ++i) {
                dstX[i] += biasZ[i];
            }
        }
    }
}

void MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ        = dst + planeNumber * 4 * z;
        const float* biasZ = bias + 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX = dstZ + 4 * p;
            for (int i = 0; i < 4; ++i) {
                dstX[i] += biasZ[i];
                if (dstX[i] < 0) {
                    dstX[i] = 0;
                }
            }
        }
    }
}

void MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ        = dst + planeNumber * 4 * z;
        const float* biasZ = bias + 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX = dstZ + 4 * p;
            for (int i = 0; i < 4; ++i) {
                dstX[i] += biasZ[i];
                if (dstX[i] < 0) {
                    dstX[i] = 0;
                }
                if (dstX[i] > 6.0f) {
                    dstX[i] = 6.0f;
                }
            }
        }
    }
}

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] = s[j];
        }
    }
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] += s[j];
        }
    }
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    for (int j = 0; j < depthQuad; j++) {
        const float* slopeZ = slope + 4 * j;
        const float* srcZ   = src + 4 * j * sizeQuad;
        float* dstZ         = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            for (int c = 0; c < 4; c++) {
                if (srcZ[4 * i + c] < 0) {
                    dstZ[4 * i + c] = srcZ[4 * i + c] * slopeZ[c];
                } else {
                    dstZ[4 * i + c] = srcZ[4 * i + c];
                }
            }
        }
    }
}
void MNNPackC4(float* dst, const float* src, size_t area, size_t depth) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const float* srcChannel[4];
    const float* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + area * + y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < 4; ++y) {
                dst[0] = srcChannel[y][0];
                srcChannel[y]++;
                dst++;
            }
        }
        srcOffset += area * 4;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dst[0] = srcChannel[y][0];
                srcChannel[y]++;
                dst++;
            }
            for(y = remain; y < 4; ++y) {
                dst[0] = 0;
                dst++;
            }
        }
    }
}

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const float* srcChannel[4];
    const float* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = srcChannel[y][0];
                srcChannel[y] += 4;
                dst++;
            }
        }
        srcOffset += area * 4;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = srcChannel[y][0];
                srcChannel[y] += 4;
                dst++;
            }
        }
    }
}

void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 6;
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    auto hP = h / 6;
    auto hR = hP * 6;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 6)*6*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 6 * l;
            auto sourceY = source + y * 6;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 6 * x, sourceY + x * h, 6 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 6 * l;
            auto sourceY = source + hP * 6;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 6 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    for (int y=0; y<h; ++y) {
        auto yR = y % 6;
        auto yC = y / 6;
        for (int x=0; x<l; ++x) {
            dest[x * 6 + yR + yC * 6 * l] = source[x + y * l];
        }
    }
}
void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    return MNNPackedMatMulRemain(C, A, B, 16, parameter, cache, postParameters, bias);
    //return _AVX_MNNPackedMatMulFMA(C, A, B, parameter, cache);
}
void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 6;
    auto hC4 = UP_DIV(h, 4);
    for (int y=0; y<hC4; ++y) {
        ::memset(C + y * cStride, 0, eSize * 4 * sizeof(float));
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
        alpha = postParameters[0];
        beta = postParameters[1];
    }
    
    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int ry=0; ry<h; ++ry) {
            auto y = ry / 4;
            auto yRemain = ry % 4;
            auto bY = B + y * bStride;
            auto dstY = dst + y * cStride;
            int wdy = ry / 6;
            int wdyRemain = ry % 6;
            auto weight = B + wdy * bStride + wdyRemain;
            float summer = 0.0f;
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * 16;
                auto wZ = weight + z * 6;
                summer += wZ[0] * aZ[0];
            }
            float originValue = dstY[yRemain];
            if (nullptr != bias) {
                originValue = bias[ry];
            }
            auto dstValue = originValue * beta + alpha * summer;
            dstValue = std::min(dstValue, maxValue);
            dstValue = std::max(dstValue, minValue);
            dstY[yRemain] = dstValue;
        }
    }
}
void MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal) {
    const int mid = 1;
    auto ePack = e / 16;
    auto eDiv = UP_DIV(e, 16);
    auto lC4 = l / 4;
    auto lDiv = UP_DIV(l, 4);
    auto eRemain = ePack * 16;
    auto lRemain = lC4 * 4;
    if (eRemain != e) {
        ::memset(dest, 0, eDiv * l * 16 * mid * sizeof(float));
    }
    for (int y=0; y<e; ++y) {
        auto yR = y % 16;
        auto yC = y / 16;
        for (int z=0; z<mid; ++z) {
            for (int x=0; x<l; ++x) {
                auto xR = x % 4;
                auto xC = x / 4;
                dest[(x * mid + z) * 16 + yR + yC * 16 * l * mid] = source[(xC * mid + z) * eReal * 4 + y * 4 + xR];
            }
        }
    }
}

void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    auto count = countC8 * 8;
    auto param = parameters[0];
    float xLimit = 87;
    for (int i = 0; i < count; ++i) {
        auto x         = -source[i];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        int div        = (x * parameters[1]);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        auto t = xReamin;
        auto expRemain =
            ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + parameters[3]) * t +
            parameters[2];
        dest[i] = expBasic * expRemain;
    }
}
#endif // no MNN_USE_SSE

void MNNMaxFloat(float* input, float* maxBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                maxBuffer[j] = std::max(input[i * UNIT * 2 + j * 2 + m], maxBuffer[j]);
            }
        }
    }
}
void MNNMinFloat(float* input, float* minBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                minBuffer[j] = std::min(input[i * UNIT * 2 + j * 2 + m], minBuffer[j]);
            }
        }
    }
}
void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * 4 * z;
        const float* srcZ   = src + planeNumber * 4 * z;
        auto biasZ = Vec4::load(bias + 4 * z);
        auto alphaZ = Vec4::load(alpha + 4 * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + 4 * p;
            const float* srcX = srcZ + 4 * p;
            Vec4::save(dstX, (Vec4::load(srcX) * alphaZ) + biasZ);
        }
    }
}



void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                       size_t dstStride, size_t srcStride) {
    dstStride /= sizeof(int16_t);
    srcStride /= sizeof(uint8_t);
    for (int z = 0; z < sizeQuad; ++z) {
        auto dstZ = dst + dstStride * z;
        auto srcZ = src + srcStride * z;
        for (int j = 0; j < 4; ++j) {
            dstZ[j] = (int16_t)((int32_t)srcZ[j] - (int32_t)zeroPoint);
        }
    }
}

void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* colAddr, const uint8_t* srcStart, size_t zeroPoint, size_t sizeQuad,
                                     size_t depthQuad, size_t dstZStep, size_t srcZStep) {
    dstZStep /= sizeof(int16_t);
    srcZStep /= sizeof(uint8_t);
    for (int sz = 0; sz < depthQuad; ++sz) {
        auto dstZ = colAddr + sz * dstZStep;
        auto srcZ = srcStart + sz * srcZStep;
        MNNUInt8ToInt16WithOffsetC4Common(dstZ, srcZ, zeroPoint, sizeQuad, 4 * sizeof(int16_t), 4 * sizeof(uint8_t));
    }
}

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size) {
    int i;
    for (i = 0; i < size; ++i) {
        if (src[i] < 0) {
            dst[i] = 0;
        } else {
            dst[i] = src[i];
        }
    }
}
void MNNPowC8(float* dest, const float* source, const float* powfParam, size_t betaInt, size_t countC8) {
    const int count          = countC8 * 8;
    const float powfConstant = powfParam[6];
    for (int i = 0; i < count; ++i) {
        float result = 1, x, xInv = 1 / source[i];
        for (int j = 0; j < betaInt; result *= xInv, ++j)
            ;
        for (x = source[i]; x >= 1.25; x /= 1.5, result *= powfConstant)
            ;
        float t = x - 1;
        float powRemain =
            powfParam[0] +
            t * (powfParam[1] + t * (powfParam[2] + t * (powfParam[3] + t * (powfParam[4] + t * powfParam[5]))));
        result *= powRemain;
        dest[i] = result;
    }
}


#endif // no MNN_USE_NEON


void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    int z, x;
    int cur = 0;
    memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(uint8_t));
    for (z = 0; z < depth; ++z) {
        int plane         = z / 4;
        uint8_t* dstPlane = plane * area * 4 + dst;
        int offset        = z % 4;
        for (x = 0; x < area; ++x) {
            dstPlane[4 * x + offset] = src[cur++];
        }
    }
}

void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    int x;
    int z;
    int cur = 0;
    for (z = 0; z < depth; ++z) {
        int plane               = z / 4;
        const uint8_t* srcPlane = plane * area * 4 + src;
        int offset              = z % 4;
        for (x = 0; x < area; ++x) {
            dst[cur++] = srcPlane[4 * x + offset];
        }
    }
}

void MNNUnpackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    if (depth == 4) {
        ::memcpy(dst, src, area * depth * sizeof(uint8_t));
        return;
    }
#ifdef MNN_USE_NEON
    if (depth == 3) {
        uint8x16x4_t rgba;
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        int staC16  = (int)area / 16;
        for (int i = 0; i < staC16; sta += 16, ++i) {
            auto rgb    = vld3q_u8(src + sta * 3);
            rgba.val[0] = rgb.val[0];
            rgba.val[1] = rgb.val[1];
            rgba.val[2] = rgb.val[2];
            vst4q_u8(dst + 4 * sta, rgba);
        }
        sta = staC16 * 16;

        for (; sta < area; ++sta) {
            auto s = src + sta * 3;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = s[1];
            d[2]   = s[2];
            d[3]   = 0;
        }

        return;
    }
    if (depth == 1) {
        uint8x16x4_t rgba;
        rgba.val[1] = vdupq_n_u8(0);
        rgba.val[2] = vdupq_n_u8(0);
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        for (; sta < area; sta += 16) {
            rgba.val[0] = vld1q_u8(src + sta);
            vst4q_u8(dst + 4 * sta, rgba);
        }

        for (; sta < area; ++sta) {
            auto s = src + sta;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = 0;
            d[2]   = 0;
            d[3]   = 0;
        }

        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = (src + hi * c);
        auto dstHeight = (dst + hi * 4);
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * area * 4 + i] = srcHeight[4 * ci + i];
            }
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + area * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 4;
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNUnpackTranspose(float* dst, const float* src, size_t area, size_t depth) {
#ifdef MNN_USE_NEON
    if (1 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 4 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld1q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue;
            dstValue.val[1] = zeroValue;
            dstValue.val[2] = zeroValue;
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[i];
            dst[4 * i + 1] = 0.0f;
            dst[4 * i + 2] = 0.0f;
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
    if (3 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 12 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld3q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue.val[0];
            dstValue.val[1] = srcValue.val[1];
            dstValue.val[2] = srcValue.val[2];
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[3 * i + 0];
            dst[4 * i + 1] = src[3 * i + 1];
            dst[4 * i + 2] = src[3 * i + 2];
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * 4;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci * area, Vec4::load(srcHeight + 4 * ci));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + area * cAlign;

#ifdef MNN_USE_NEON
    auto zeroVector = vdupq_n_f32(0.0f);
#endif

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * c;
        float* dstHeight       = dstAlign + hi * 4;
#ifdef MNN_USE_NEON
        vst1q_f32(dstHeight, zeroVector);
#else
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
#endif
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    if (1 == area) {
        ::memcpy(dst, src, depth * sizeof(uint8_t));
        return;
    }
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    if (cAlign == c) {
        int32_t* dst32       = (int32_t*)dst;
        const int32_t* src32 = (int32_t*)src;
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src32 + hi;
            auto dstHeight = dst32 + hi * cDiv4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci] = srcHeight[ci * area];
            }
        }
        return;
    }

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 4;
        auto dstHeight = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * area + i];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 4;
        auto dstHeight = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTranspose(float* dst, const float* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * 4;
        float* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci, Vec4::load(srcHeight + 4 * ci * area));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * 4;
        float* dstHeight       = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNRelu6(float* dst, const float* src, size_t size) {
    int i;
    for (i = 0; i < size; ++i) {
        if (src[i] < 0) {
            dst[i] = 0;
        } else {
            dst[i] = src[i] < 6 ? src[i] : 6;
        }
    }
}

void MNNExp(float* dst, const float* src, size_t dataSize) {
    int countC8        = (int)dataSize / 8;
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        static float parameters[] = {
            (float)log(2.0f), 1.0f / (float)log(2.0f), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
        MNNExpC8(dst, src, parameters, countC8);
    }
    int remain = countC8 * 8;
    auto param = log(2.0f);
    float xLimit = 87;
    for (int i = remain; i < dataSize; i++) {
        /*Origin Function*/
        //dst[i] = expf(-src[i]);
        /*Approciate Function*/
        
        auto x         = -src[i];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        
        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        
        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dst[i]  = expBasic * expRemain;
    }
}

// Lambert's series with 7 divisions
// reference from
// https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
inline float tanhf_poly(float value) {
    if (value > 5.0) {
        return 1.0;
    } else if (value <= -5.0) {
        return -1.0;
    } else {
        float x2 = value * value;
        float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }
}
void MNNTanh(float* dst, const float* src, size_t dataSize) {
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        dst[i] = tanhf_poly(src[i]);
    }
}

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope) {
    float slopeValue[4];
    for (int i=0; i<4; ++i) {
        slopeValue[i] = slope;
    }
    MNNReluWithSlopeChannel(dst, src, slopeValue, sizeQuad, 1);
}
void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope) {
    int sizeQuad = size / 4;
    int start = 0;
    if (sizeQuad > 0) {
        MNNReluWithSlope(dst, src, sizeQuad, slope);
        start = sizeQuad * 4;
    }
    for (int j = start; j < size; j++) {
        if (src[j] < 0) {
            dst[j] = src[j] * slope;
        } else {
            dst[j] = src[j];
        }
    }
}

void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number) {
    int numberC4 = (int)number / 4;
    int start = 0;
    if (numberC4 > 0) {
        float biasC4[4] = {
            bias,
            bias,
            bias,
            bias
        };
        float alphaC4[4] = {
            alpha,
            alpha,
            alpha,
            alpha
        };
        MNNScaleAndAddBias(dst, src, biasC4, alphaC4, numberC4, 1);
        start = numberC4 * 4;
    }
    for (int i=start; i<number; ++i) {
        dst[i] = src[i] * alpha + bias;
    }
}
void MNNAxByClamp(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height, const float* parameters) {
    int widthC4 = (int)width / 4;
    if (widthC4 > 0) {
        auto minF = Vec4(parameters[2]);
        auto maxF = Vec4(parameters[3]);
        auto alpha = Vec4(parameters[0]);
        auto beta = Vec4(parameters[1]);
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                auto av = Vec4::load(a + 4 * x);
                auto bv = Vec4::load(b + 4 * x);
                auto cv = av * alpha + bv * beta;
                cv = Vec4::min(cv, maxF);
                cv = Vec4::max(cv, minF);
                Vec4::save(c + 4 * x, cv);
            }
        }
        width = width - 4*widthC4;
        C = C + widthC4 * 4;
        A = A + widthC4 * 4;
        B = B + widthC4 * 4;
    }
    if (width > 0) {
        auto minF = parameters[2];
        auto maxF = parameters[3];
        auto alpha = parameters[0];
        auto beta = parameters[1];
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                auto av = a[x];
                auto bv = b[x];
                auto cv = av * alpha + bv * beta;
                cv = std::min(cv, maxF);
                cv = std::max(cv, minF);
                c[x] = cv;
            }
        }
    }
}
#ifndef MNN_USE_NEON
void MNNAxByClampBroadcastC4(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = Vec4(parameters[2]);
    auto maxF = Vec4(parameters[3]);
    auto beta = Vec4(parameters[1]);
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + 4 * y;
        auto bv = Vec4::load(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = Vec4::load(a + 4 * x);
            auto cv = av + bv * beta;
            cv = Vec4::min(cv, maxF);
            cv = Vec4::max(cv, minF);
            Vec4::save(c + 4 * x, cv);
        }
    }
}

void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    float maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

void MNNVectorTop1Int32(int32_t* input, int32_t* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    int32_t maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

#endif

#ifndef MNN_USE_SSE

void MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            Vec4 sumValue = Vec4(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = sumValue + Vec4::load(A + x * 4) * Vec4::load(by + x * 4);
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            C[y] = sumRemain + sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3];
        }
    } else {
        auto hC4 = h / 4;
        auto hR = hC4 * 4;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 4 * y;
            Vec4 sumValue = Vec4(0.0f);
            if (biasPtr != nullptr) {
                sumValue = Vec4::load(biasPtr + 4 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + Vec4(A[x]) * Vec4::load(bs + h * x);
            }
            Vec4::save(C + 4 * y, sumValue);
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
#endif
