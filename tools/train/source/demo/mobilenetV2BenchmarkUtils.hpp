//
//  MobilenetV2BenchmarkUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobilenetV2BenchmarkUtils_hpp
#define MobilenetV2BenchmarkUtils_hpp

#include <MNN/expr/Module.hpp>
#include <string>

class MobilenetV2BenchmarkUtils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, const int numClasses, const int addToLabel,
                      std::string trainImagesFolder, std::string trainImagesTxt,
                      std::string testImagesFolder, std::string testImagesTxt, const int BatchSize,
                      const int trainQuantDelayEpoch = 10, const int quantBits = 8);
};

#endif
