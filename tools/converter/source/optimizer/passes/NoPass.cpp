#include "converter/source/optimizer/passes/PassRegistry.hpp"
#include "converter/source/optimizer/passes/Pass.hpp"

namespace MNN {
namespace passes {

REGISTER_REWRITE_PASS(NoPass)                            \
    .Verify([](PassContext* context) { return false; })   \
    .Rewrite([](PassContext* context) { return false; });

}  // namespace passes
}  // namespace MNN
