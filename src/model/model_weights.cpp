#include "model/model_weights.h"

namespace mini_llm::model {

bool ModelWeights::load_from_manifest(const std::string& manifest_path) {
    // TODO: 解析 manifest 并加载对应权重资源
    (void)manifest_path;
    return false;
}

bool ModelWeights::ready() const {
    // TODO: 根据加载状态返回真实结果
    return false;
}

} // namespace mini_llm::model
