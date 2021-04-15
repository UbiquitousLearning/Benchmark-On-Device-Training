#include <string>
#include <unordered_map>
#include <mutex>

#include "MNN/MNNDefine.h"
#include "converter/source/optimizer/passes/PassRegistry.hpp"

namespace MNN {
namespace passes {

// All registered passes.
static std::unordered_map<std::string, std::unique_ptr<Pass>>*
    AllRegisteredPasses() {
    static std::unordered_map<std::string, \
                              std::unique_ptr<Pass>> g_registered_passes;
    return &g_registered_passes;
}
// All registered pass managers.
static std::vector<std::unique_ptr<PassManager>>* AllRegisteredPassManagers() {
    static std::vector<std::unique_ptr<PassManager>> g_registered_pass_managers;
    return &g_registered_pass_managers;
}

static std::mutex g_mutex;

/*static*/ PassManager* PassManagerRegistry::GetPassManager(int index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto* g_registered_pass_managers = AllRegisteredPassManagers();
    MNN_CHECK(index < g_registered_pass_managers->size(),
              "The pass manager index is out of bounds.");
    return (*g_registered_pass_managers)[index].get();
}

/*static*/ std::vector<PassManager*> PassManagerRegistry::GetAllPassManagers() {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<PassManager*> pass_managers;
    for (auto& pm : *(AllRegisteredPassManagers())) {
        pass_managers.push_back(pm.get());
    }
    return pass_managers;
}

/*static*/ void PassManagerRegistry::AddPassManager(const PassManager& pm) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto* g_registered_pass_managers = AllRegisteredPassManagers();
    g_registered_pass_managers->emplace_back(new PassManager(pm));
}

/*static*/ void PassRegistry::AddPass(std::unique_ptr<Pass>&& pass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto* g_registered_passes = AllRegisteredPasses();
    g_registered_passes->emplace(pass->name(), std::move(pass));
}

/*static*/ Pass* PassRegistry::GetPass(const std::string& pass_name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto* g_registered_passes = AllRegisteredPasses();
    const auto& it = g_registered_passes->find(pass_name);
    if (it != g_registered_passes->end()) {
        return it->second.get();
    }
    return nullptr;
}

RewritePassRegistry::RewritePassRegistry(const std::string& pass_name)
    : pass_name_(pass_name) {
    std::unique_ptr<Pass> pass(new RewritePass(pass_name));
    PassRegistry::AddPass(std::move(pass));
}

RewritePass* GetRewritePassByName(const std::string& pass_name) {
    Pass* pass = PassRegistry::GetPass(pass_name);
    MNN_CHECK(pass, "Pass has not been setup.");
    RewritePass *rewrite_pass = dynamic_cast<RewritePass*>(pass);
    if (!rewrite_pass) {
        MNN_ERROR("Pass %s is registered but not rewrite pass.",
                  pass_name.c_str());
    }
    return rewrite_pass;
}

void RewritePassRegistry::SetVerify(RewritePassRegistry::FuncType verify_fn) {
    auto *rewrite_pass = GetRewritePassByName(pass_name_);
    rewrite_pass->SetVerify(verify_fn);
}

void RewritePassRegistry::SetRewrite(RewritePassRegistry::FuncType rewrite_fn) {
    auto *rewrite_pass = GetRewritePassByName(pass_name_);
    rewrite_pass->SetRewrite(rewrite_fn);
}

}  // namespace passes
}  // namespace MNN
