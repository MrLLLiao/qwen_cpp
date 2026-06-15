#include "model/rms_norm.h"

#include <cmath>
#include <stdexcept>

namespace mini_llm::model {

RMSNorm::RMSNorm(RMSNormConfig config)
    : config_(config)
{
    if (config_.epsilon <= 0.0f)
    {
        throw std::invalid_argument("RMSNorm epsilon must be greater than 0");
    }
}

Tensor RMSNorm::forward(const Tensor& input, const Tensor& weight) const
{
    if (input.empty())
    {
        return Tensor{};
    }
    if (weight.rows() != 1 || weight.cols() != input.cols())
    {
        throw std::invalid_argument("RMSNorm weight must be [1, hidden_size]");
    }

    Tensor output(input.rows(), input.cols(), 0.0f);
    for (std::size_t r = 0; r < input.rows(); ++r)
    {
        const float* in = input.row_data(r);
        float sum_sq = 0.0f;
        for (std::size_t c = 0; c < input.cols(); ++c)
        {
            sum_sq += in[c] * in[c];
        }

        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(input.cols()) + config_.epsilon);
        float* out = output.row_data(r);
        for (std::size_t c = 0; c < input.cols(); ++c)
        {
            out[c] = in[c] * inv_rms * weight(0, c);
        }
    }
    return output;
}

const RMSNormConfig& RMSNorm::config() const
{
    return config_;
}

Tensor rms_norm(const Tensor& input,
                  const Tensor& weight,
                  float epsilon)
{
    return RMSNorm(RMSNormConfig{epsilon}).forward(input, weight);
}

} // namespace mini_llm::model
