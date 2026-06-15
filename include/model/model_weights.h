#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tensor.h"
#include "model/model_config.h"

namespace mini_llm::model {

class ModelWeights {
public:
    struct LayerWeights {
        Tensor attention_wq;
        Tensor attention_wk;
        Tensor attention_wv;
        Tensor attention_wo;
        Tensor mlp_gate;
        Tensor mlp_up;
        Tensor mlp_down;
        Tensor rms_attn_weight;
        Tensor rms_ffn_weight;
    };

    bool load_from_manifest(const std::string& manifest_path);
    void reset(ModelConfig config);

    [[nodiscard]] bool empty() const;
    bool ready() const;
    [[nodiscard]] const ModelConfig& config() const;
    [[nodiscard]] const LayerWeights& layer(std::size_t layer_id) const;
    [[nodiscard]] LayerWeights& layer(std::size_t layer_id);
    [[nodiscard]] const Tensor& token_embedding() const;
    [[nodiscard]] Tensor& token_embedding();
    [[nodiscard]] const Tensor& output_embedding() const;
    [[nodiscard]] Tensor& output_embedding();

    void set_layer_weights(std::size_t layer_id, LayerWeights weights);

private:
    [[nodiscard]] bool has_model_shape() const;

private:
    ModelConfig config_{};
    Tensor token_embedding_{};
    Tensor output_embedding_{};
    std::vector<LayerWeights> layers_{};
};

} // namespace mini_llm::model
