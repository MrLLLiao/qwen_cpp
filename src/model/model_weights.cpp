#include "model/model_weights.h"

#include <stdexcept>

namespace mini_llm::model {

bool ModelWeights::load_from_manifest(const std::string& manifest_path)
{
    (void)manifest_path;
    return false;
}

void ModelWeights::reset(ModelConfig config)
{
    if (!config.valid())
    {
        throw std::invalid_argument("ModelWeights::reset invalid model config");
    }

    config_ = config;
    token_embedding_ = Tensor(config.vocab_size, config.hidden_size, 0.0f);
    output_embedding_ = Tensor(config.hidden_size, config.vocab_size, 0.0f);
    layers_.assign(config.num_hidden_layers, LayerWeights{});
}

bool ModelWeights::empty() const
{
    return layers_.empty() || token_embedding_.empty();
}

bool ModelWeights::ready() const
{
    if (!config_.valid() || token_embedding_.empty() || output_embedding_.empty())
    {
        return false;
    }
    if (layers_.size() != config_.num_hidden_layers)
    {
        return false;
    }
    return has_model_shape();
}

const ModelConfig& ModelWeights::config() const
{
    return config_;
}

const ModelWeights::LayerWeights& ModelWeights::layer(std::size_t layer_id) const
{
    if (layer_id >= layers_.size())
    {
        throw std::out_of_range("ModelWeights::layer layer_id out of range");
    }
    return layers_[layer_id];
}

ModelWeights::LayerWeights& ModelWeights::layer(std::size_t layer_id)
{
    if (layer_id >= layers_.size())
    {
        throw std::out_of_range("ModelWeights::layer layer_id out of range");
    }
    return layers_[layer_id];
}

const Tensor& ModelWeights::token_embedding() const
{
    return token_embedding_;
}

Tensor& ModelWeights::token_embedding()
{
    return token_embedding_;
}

const Tensor& ModelWeights::output_embedding() const
{
    return output_embedding_;
}

Tensor& ModelWeights::output_embedding()
{
    return output_embedding_;
}

void ModelWeights::set_layer_weights(std::size_t layer_id, LayerWeights weights)
{
    if (layer_id >= layers_.size())
    {
        throw std::out_of_range("ModelWeights::set_layer_weights layer_id out of range");
    }
    layers_[layer_id] = std::move(weights);
}

bool ModelWeights::has_model_shape() const
{
    if (token_embedding_.rows() != config_.vocab_size || token_embedding_.cols() != config_.hidden_size)
    {
        return false;
    }
    if (output_embedding_.rows() != config_.hidden_size || output_embedding_.cols() != config_.vocab_size)
    {
        return false;
    }

    for (const auto& layer : layers_)
    {
        const auto hidden = config_.hidden_size;
        const auto inter = config_.intermediate_size;
        if (layer.attention_wq.rows() != hidden || layer.attention_wq.cols() != hidden)
        {
            return false;
        }
        if (layer.attention_wk.rows() != hidden || layer.attention_wk.cols() != config_.kv_hidden_size())
        {
            return false;
        }
        if (layer.attention_wv.rows() != hidden || layer.attention_wv.cols() != config_.kv_hidden_size())
        {
            return false;
        }
        if (layer.attention_wo.rows() != hidden || layer.attention_wo.cols() != hidden)
        {
            return false;
        }
        if (layer.mlp_gate.rows() != hidden || layer.mlp_gate.cols() != inter)
        {
            return false;
        }
        if (layer.mlp_up.rows() != hidden || layer.mlp_up.cols() != inter)
        {
            return false;
        }
        if (layer.mlp_down.rows() != inter || layer.mlp_down.cols() != hidden)
        {
            return false;
        }
        if (layer.rms_attn_weight.rows() != 1 || layer.rms_attn_weight.cols() != hidden)
        {
            return false;
        }
        if (layer.rms_ffn_weight.rows() != 1 || layer.rms_ffn_weight.cols() != hidden)
        {
            return false;
        }
    }

    return true;
}

} // namespace mini_llm::model
