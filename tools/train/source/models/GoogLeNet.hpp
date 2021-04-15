//
//  GoogLenet.hpp
//  MNN
//
//  Created by CDQ on 2021/03/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GoogLenetModels_hpp
#define GoogLenetModels_hpp

#include <MNN/expr/Module.hpp>

namespace MNN {
namespace Train { 
namespace Model {

class MNN_PUBLIC GoogLenet : public Express::Module {
public:

    GoogLenet();

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> conv3;
    std::shared_ptr<Module> incep1;
    std::shared_ptr<Module> incep2;
    std::shared_ptr<Module> incep3;
    std::shared_ptr<Module> incep4;
    std::shared_ptr<Module> incep5;
    std::shared_ptr<Module> incep6;
    std::shared_ptr<Module> incep7;
    std::shared_ptr<Module> incep8;
    std::shared_ptr<Module> incep9;

};

} // namespace Model
} // namespace Train
} // namespace MNN









#endif // GoogLenetModels_hpp