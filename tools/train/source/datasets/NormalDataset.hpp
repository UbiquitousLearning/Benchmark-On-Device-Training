//
//  NormalDataset.hpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NormalDataset_hpp
#define NormalDataset_hpp

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"

namespace MNN {
namespace Train {
class MNN_PUBLIC NormalDataset : public Dataset {
public:
    enum Mode { TRAIN, TEST };

    Example get(size_t index) override;

    size_t size() override;

    const VARP images();

    const VARP labels();

    static DatasetPtr create(const std::string path, Mode mode = Mode::TRAIN);
private:
    explicit NormalDataset(const std::string path, Mode mode = Mode::TRAIN);
    VARP mImages, mLabels;
    const uint8_t* mImagePtr  = nullptr;
    const uint8_t* mLabelsPtr = nullptr;
};
}
}


#endif // NormalDataset_hpp
