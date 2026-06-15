#pragma once

#include <vector>

#include "model/model_config.h"
#include "model/model_weights.h"
#include "model/transformer_block.h"
#include "tensor.h"

namespace mini_llm::model {

class QwenModel final {
public:
    QwenModel() = default;
    explicit QwenModel(ModelWeights weights);

    void load_weights(ModelWeights weights);
    [[nodiscard]] bool ready() const;
    [[nodiscard]] const ModelConfig& config() const;

    [[nodiscard]] Tensor forward(const Tensor& token_embeddings,
                                   const Tensor* additive_mask = nullptr) const;
    [[nodiscard]] Tensor forward_embeddings(const Tensor& token_embeddings,
                                              const Tensor* additive_mask = nullptr) const;
    [[nodiscard]] Tensor logits(const Tensor& hidden_states) const;

private:
    void rebuild_blocks();

private:
    ModelWeights weights_{};
    std::vector<TransformerBlock> blocks_{};
};

} // namespace mini_llm::model
