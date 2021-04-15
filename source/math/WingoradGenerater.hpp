//
//  WingoradGenerater.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WingoradGenerater_hpp
#define WingoradGenerater_hpp
#include <memory>
#include "math/Matrix.hpp"
namespace MNN {
namespace Math {
class MNN_PUBLIC WinogradGenerater {
public:
    // If dividedInG, make A, B not frac, else make A, G not frac
    WinogradGenerater(int computeUnit, int kernelSize, float interp = 0.5f, bool dividedInG = false);
    ~WinogradGenerater() = default;

    std::shared_ptr<Tensor> A() const {
        return mA;
    }
    std::shared_ptr<Tensor> B() const {
        return mB;
    }
    std::shared_ptr<Tensor> G() const {
        return mG;
    }

    std::shared_ptr<Tensor> allocTransformWeight(const Tensor* originWeight, int unitCi = 4, int unitCo = 4, bool alloc = true);
    void transformWeight(const Tensor* dest, const Tensor* source);

private:
    std::shared_ptr<Tensor> mA;
    std::shared_ptr<Tensor> mG;
    std::shared_ptr<Tensor> mB;
    int mUnit;
    int mKernelSize;
};
} // namespace Math
} // namespace MNN

#endif /* WingoradGenerater_hpp */
