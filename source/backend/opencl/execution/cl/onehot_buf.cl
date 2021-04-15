#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void onehot_buf(__global FLOAT *indices, __global FLOAT *depthPtr,
                     __private const int outerSize, __global FLOAT *onValuePtr,
                     __global FLOAT *offValuePtr, __global FLOAT *outputPtr) {
  int i = get_global_id(0);
  int depth = (int)*depthPtr;
  FLOAT onValue = *onValuePtr;
  FLOAT offValue = *offValuePtr;

  if (i < outerSize) {
      for(int j = 0; j < depth; j++) {
          if(j == (int)indices[i]) {
              outputPtr[i * depth + j] = onValue;
          } else {
              outputPtr[i * depth + j] = offValue;
          }
      }
  }
} 