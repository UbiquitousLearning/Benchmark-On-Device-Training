//
//  MetalReLU.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void relu6_x1(const device ftype *in  [[buffer(0)]],
                     device ftype *out       [[buffer(1)]],
                     constant float4 &minMax   [[buffer(2)]],
                     uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = clamp(in[int(gid)], (ftype)(minMax.x), (ftype)(minMax.y));
}

kernel void relu6_x4(const device ftype4 *in [[buffer(0)]],
                     device ftype4 *out      [[buffer(1)]],
                     constant float4 &minMax   [[buffer(2)]],
                     uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = clamp(in[int(gid)], (ftype4)minMax.x, (ftype4)minMax.y);
}
