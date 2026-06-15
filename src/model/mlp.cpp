#include "model/mlp.h"

#include <cmath>
#include <stdexcept>

#include "ops/matmul.h"

namespace
{
[[nodiscard]] float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}
}

namespace mini_llm::model {

MLP::MLP(MLPConfig config)
    : config_(config)
{
    if (config_.hidden_size == 0 || config_.intermediate_size == 0)
    {
        throw std::invalid_argument("MLP config invalid");
    }
}

void MLP::set_weights(const Tensor& gate, const Tensor& up, const Tensor& down)
{
    gate_ = gate;
    up_ = up;
    down_ = down;
}

Tensor MLP::forward(const Tensor& input) const
{
    if (!valid_weights())
    {
        throw std::logic_error("MLP weights are not ready");
    }
    if (input.cols() != config_.hidden_size)
    {
        throw std::invalid_argument("MLP input hidden size mismatch");
    }

    const Tensor gate_proj = matmul(input, gate_);
    const Tensor up_proj = matmul(input, up_);
    Tensor activated(input.rows(), config_.intermediate_size, 0.0f);
    for (std::size_t r = 0; r < input.rows(); ++r)
    {
        float* out = activated.row_data(r);
        const float* g = gate_proj.row_data(r);
        const float* u = up_proj.row_data(r);
        for (std::size_t c = 0; c < config_.intermediate_size; ++c)
        {
            out[c] = silu(g[c]) * u[c];
        }
    }

    return matmul(activated, down_);
}

const MLPConfig& MLP::config() const
{
    return config_;
}

bool MLP::valid_weights() const
{
    return gate_.rows() == config_.hidden_size && gate_.cols() == config_.intermediate_size
        && up_.rows() == config_.hidden_size && up_.cols() == config_.intermediate_size
        && down_.rows() == config_.intermediate_size && down_.cols() == config_.hidden_size;
}

} // namespace mini_llm::model
