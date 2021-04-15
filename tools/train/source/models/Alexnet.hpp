//
//  Alexnet.hpp
//  MNN
//
//  Created by CDQ on 2021/02/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AlexnetModels_hpp
#define AlexnetModels_hpp

#include <MNN/expr/Module.hpp>

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC Alexnet : public Express::Module {
public:
    Alexnet();

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;

    std::shared_ptr<Express::Module> conv1;
    std::shared_ptr<Express::Module> conv2;
    std::shared_ptr<Express::Module> conv3;
    std::shared_ptr<Express::Module> conv4;
    std::shared_ptr<Express::Module> conv5;
    std::shared_ptr<Express::Module> ip1;
    std::shared_ptr<Express::Module> ip2;
    std::shared_ptr<Express::Module> dropout1;
    std::shared_ptr<Express::Module> dropout2;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // AlexnetModels_hpp
