#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdint.h>
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "backend/cpu/compute/CommonOptFunction.h"

#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)
#endif

// ========= CommonOptFunction.cpp ===========
extern "C" {
void _AVX512_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose);
void _AVX512_MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal);
void _AVX512_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias);
void _AVX512_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias);
void _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);

}
