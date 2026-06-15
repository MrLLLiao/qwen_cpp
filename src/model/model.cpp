#include "model/model.h"

#include <stdexcept>

#include "model/rms_norm.h"
#include "ops/matmul.h"

namespace mini_llm::model {

QwenModel::QwenModel(ModelWeights weights)
{
    load_weights(std::move(weights));
}

void QwenModel::load_weights(ModelWeights weights)
{
    if (!weights.ready())
    {
        throw std::invalid_argument("QwenModel::load_weights weights not ready");
    }

    weights_ = std::move(weights);
    rebuild_blocks();
}

bool QwenModel::ready() const
{
    return weights_.ready() && blocks_.size() == weights_.config().num_hidden_layers;
}

const ModelConfig& QwenModel::config() const
{
    return weights_.config();
}

Tensor QwenModel::forward(const Tensor& token_embeddings,
                            const Tensor* additive_mask) const
{
    return logits(forward_embeddings(token_embeddings, additive_mask));
}

Tensor QwenModel::forward_embeddings(const Tensor& token_embeddings,
                                       const Tensor* additive_mask) const
{
    if (!ready())
    {
        throw std::logic_error("QwenModel::forward_embeddings model is not ready");
    }

    if (token_embeddings.cols() != config().hidden_size)
    {
        throw std::invalid_argument("QwenModel::forward_embeddings hidden size mismatch");
    }

    Tensor hidden = token_embeddings;
    for (const auto& block : blocks_)
    {
        hidden = block.forward(hidden, additive_mask);
    }
    return hidden;
}

Tensor QwenModel::logits(const Tensor& hidden_states) const
{
    if (!weights_.ready())
    {
        throw std::logic_error("QwenModel::logits model is not ready");
    }
    if (hidden_states.cols() != config().hidden_size)
    {
        throw std::invalid_argument("QwenModel::logits hidden size mismatch");
    }
    return matmul(hidden_states, weights_.output_embedding());
}

void QwenModel::rebuild_blocks()
{
    blocks_.clear();
    blocks_.reserve(weights_.config().num_hidden_layers);
    for (std::size_t layer = 0; layer < weights_.config().num_hidden_layers; ++layer)
    {
        TransformerBlock block(layer, weights_.config());
        block.set_weights(weights_.layer(layer));
        blocks_.push_back(std::move(block));
    }
}

} // namespace mini_llm::model
