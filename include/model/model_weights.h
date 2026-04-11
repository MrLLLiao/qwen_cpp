#pragma once

#include <string>

namespace mini_llm::model {

class ModelWeights {
public:
    // TODO: 定义权重加载入口（本地路径、manifest、后端适配）
    bool load_from_manifest(const std::string& manifest_path);

    // TODO: 暴露权重可用性检查与版本信息接口
    bool ready() const;
};

} // namespace mini_llm::model
