//
//  Squeezenet.hpp
//  MNN
//
//  Created by CDQ on 2021/03/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SqueezenetModels_hpp
#define SqueezenetModels_hpp

#include <MNN/expr/Module.hpp>

namespace MNN {
namespace Train { 
namespace Model {

class MNN_PUBLIC Squeezenet : public Express::Module {
public:

    Squeezenet();

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> fire1;
    std::shared_ptr<Module> fire2;
    std::shared_ptr<Module> fire3;
    std::shared_ptr<Module> fire4;
    std::shared_ptr<Module> fire5;
    std::shared_ptr<Module> fire6;
    std::shared_ptr<Module> fire7;
    std::shared_ptr<Module> fire8;

};

} // namespace Model
} // namespace Train
} // namespace MNN









#endif // SqueezenetModels_hpp