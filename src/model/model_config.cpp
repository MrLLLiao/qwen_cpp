#include "model/model_config.h"

#include <stdexcept>

namespace mini_llm::model {

bool ModelConfig::valid() const
{
    if (vocab_size == 0 || hidden_size == 0 || num_hidden_layers == 0 || num_attention_heads == 0)
    {
        return false;
    }
    if (num_key_value_heads == 0 || intermediate_size == 0 || max_position_embeddings == 0)
    {
        return false;
    }
    if ((hidden_size % num_attention_heads) != 0)
    {
        return false;
    }
    if ((num_attention_heads % num_key_value_heads) != 0)
    {
        return false;
    }
    return rms_norm_eps > 0.0f && rope_theta > 0.0f && rope_scale > 0.0f;
}

std::size_t ModelConfig::head_dim() const
{
    if (num_attention_heads == 0 || (hidden_size % num_attention_heads) != 0)
    {
        throw std::logic_error("ModelConfig::head_dim invalid attention head config");
    }
    return hidden_size / num_attention_heads;
}

std::size_t ModelConfig::kv_hidden_size() const
{
    return num_key_value_heads * head_dim();
}

} // namespace mini_llm::model
