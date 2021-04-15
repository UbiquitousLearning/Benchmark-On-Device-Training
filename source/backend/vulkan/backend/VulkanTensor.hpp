//
//  VulkanTensor.hpp
//  MNN
//
//  Created by MNN on 2020/03/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanTensor_hpp
#define VulkanTensor_hpp
#include <MNN/Tensor.hpp>
#include "core/NonCopyable.hpp"
#include "VulkanImage.hpp"
#include "VulkanBuffer.hpp"
namespace MNN {
class VulkanTensor : public NonCopyable {
public:
    ~VulkanTensor() {
    }
    VulkanTensor(const Tensor* shape, const VulkanMemoryPool& pool, const VkPhysicalDeviceLimits& limits, bool seperate = false);
    void release();

    size_t imageSize() const {
        return mImage.size();
    }
    const std::vector<int>& blocks() const {
        return mBlocks;
    }
    const VulkanImage* image(int index = 0) const {
        return mImage[index].get();
    }
    static int getAlignSize(const Tensor* tensor);
private:
    std::vector<std::shared_ptr<VulkanImage>> mImage;
    std::vector<int> mBlocks;
};
}
#endif
