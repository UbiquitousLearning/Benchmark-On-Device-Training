//
//  MnistBenchmarkUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MnistBenchmarkUtils_hpp
#define MnistBenchmarkUtils_hpp
#include <MNN/expr/Module.hpp>
class MnistBenchmarkUtils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, std::string root, int BatchSize, std::string NetName);
};
#endif
