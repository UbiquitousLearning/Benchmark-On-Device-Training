#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void
onehot(__read_only image2d_t indices, __read_only image2d_t depthPtr,
       __private const int outerSize, __read_only image2d_t onValuePtr,
       __read_only image2d_t offValuePtr, __write_only image2d_t outputPtr) {
  int i = get_global_id(0);
  float4 depth = read_imagef(depthPtr, SAMPLER, (int2)(0, 0));
  float4 onValue = read_imagef(onValuePtr, SAMPLER, (int2)(0, 0));
  float4 offValue = read_imagef(offValuePtr, SAMPLER, (int2)(0, 0));

  if (i < outerSize) {
    float4 indice = read_imagef(indices, SAMPLER, (int2)(i / 4, 0));
    float index;
    if (i % 4 == 0) {
      index = indice.x;
    } else if (i % 4 == 1) {
      index = indice.y;
    } else if (i % 4 == 2) {
      index = indice.z;
    } else if (i % 4 == 3) {
      index = indice.w;
    }
    // printf("i: %d, %f\n", i, index);
    for (int j = 0; j < (int)depth.x; j += 4) {
      float4 result;
      if (j < (int)depth.x) {
        if (j == (int)index) {
          result.x = onValue.x;
        } else {
          result.x = offValue.x;
        }
      }
      if (j + 1 < (int)depth.x) {
        if (j + 1 == (int)index) {
          result.y = onValue.x;
        } else {
          result.y = offValue.x;
        }
      }
      if (j + 2 < (int)depth.x) {
        if (j + 2 == (int)index) {
          result.z = onValue.x;
        } else {
          result.z = offValue.x;
        }
      }
      if (j + 3 < (int)depth.x) {
        if (j + 3 == (int)index) {
          result.w = onValue.x;
        } else {
          result.w = offValue.x;
        }
      }
      write_imagef(outputPtr, (int2)(j / 4, i), result);
    }
  }
} 