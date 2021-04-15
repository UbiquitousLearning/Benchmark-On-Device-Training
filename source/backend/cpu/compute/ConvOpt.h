//
//  ConvOpt.h
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvOpt_h
#define ConvOpt_h

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONVOLUVTION_RUN_BASIC(l, t, r, b, TYPE, alpha)                                                               \
    for (dy = t; dy < b; ++dy) {                                                                                      \
        int srcStartY      = dy * strideY - padY;                                                                     \
        float* dst_y       = dst_z + width * 4 * dy;                                                                  \
        const TYPE* src_dy = srcOrigin + srcStartY * src_width * 4;                                                   \
        int sfy            = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));                                                \
        int efy            = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));                          \
        for (dx = l; dx < r; ++dx) {                                                                                  \
            int srcStartX            = dx * strideX - padX;                                                           \
            const TYPE* src_dx       = src_dy + 4 * srcStartX;                                                        \
            float* dst_x             = dst_y + 4 * dx;                                                                \
            int sfx                  = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));                                      \
            int efx                  = ALIMIN(kernel_width, UP_DIV(src_width - srcStartX, dilateX));                  \
            const TYPE* src_unit     = src_dx + (sfx * dilateX_step + sfy * dilateY_step);                            \
            const TYPE* weight_start = weight_dz + (16 * sfx + weight_sy_step * sfy);                                 \
            MNNConvSlideWindowBorder(dst_x, src_unit, weight_start, src_depth_quad, src_z_step, efx - sfx, efy - sfy, \
                                     weight_sy_step, weight_sz_step, dilateX_step, dilateY_step, alpha);              \
        }                                                                                                             \
    }


void MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
void MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                size_t srcHStep, size_t dstHStep);

void MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
void MNNDeconvRunForLineDepthwise(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                  size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step);

void MNNGemmFloatUnit_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, size_t weight_depth_offset);

void MNNGemmFloatOne_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                       size_t dst_depth_quad, size_t weight_depth_offset);
void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset);
void MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height);
void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height);
void MNNMatrixMax(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height);
void MNNMatrixProd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                   size_t bStride, size_t height);

void MNNMatrixAddCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height);
void MNNMatrixSubCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height);
void MNNMatrixMaxCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height);

void MNNMatrixProdCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height);

int MNNGetConvolutionTileNumber();

#ifdef __cplusplus
}
#endif

#endif /* ConvOpt_h */
