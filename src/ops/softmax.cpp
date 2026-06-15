#include "ops/softmax.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

Softmax::Softmax(SoftmaxConfig config) : config_(config) {}

static bool is_valid_tensor_matrix(const Tensor& input)
{
    return input.rows() > 0 && input.cols() > 0;
}

static Tensor forward_row(const Tensor& input, const SoftmaxConfig& config)
{
    if (!is_valid_tensor_matrix(input))
    {
        return {0, 0, 0.0f};
    }

    Tensor output(input.rows(), input.cols());
    std::vector<double> expValues(input.size(), 0.0);

    const float temperature = config.temperature;
    const float epsilon = config.epsilon;

    for (size_t row = 0; row < input.rows(); ++ row)
    {
        double rowMax = input.at(row, 0);
        for (size_t col = 1; col < input.cols(); ++col)
        {
            rowMax = std::max(rowMax, static_cast<double>(input.at(row, col)));
        }

        double sumExp = 0.0;
        for (size_t col = 0; col < input.cols(); ++col)
        {
            const float x = input.at(row, col);
            const double expValue = std::exp((x - rowMax) / temperature);
            expValues[row * input.cols() + col] = expValue;
            sumExp += expValue;
        }

        sumExp += epsilon;

        for (size_t col = 0; col < input.cols(); ++col)
        {
            output.at(row, col) = static_cast<float>(expValues[row * input.cols() + col] / sumExp);
        }
    }

    return output;
}

static Tensor forward_col(const Tensor& input, const SoftmaxConfig& config)
{
    if (!is_valid_tensor_matrix(input))
    {
        return {0, 0, 0.0f};
    }

    Tensor output(input.rows(), input.cols());
    std::vector<double> expValues(input.size(), 0.0);

    const float temperature = config.temperature;
    const float epsilon = config.epsilon;

    for (size_t col = 0; col < input.cols(); ++col)
    {
        double colMax = input.at(0, col);
        for (size_t row = 1; row < input.rows(); ++row)
        {
            colMax = std::max(colMax, static_cast<double>(input.at(row, col)));
        }

        double sumExp = 0.0;
        for (size_t row = 0; row < input.rows(); ++row)
        {
            const float x = input.at(row, col);
            const double expValue = std::exp((x - colMax) / temperature);
            expValues[row * input.cols() + col] = expValue;
            sumExp += expValue;
        }

        sumExp += epsilon;

        for (size_t row = 0; row < input.rows(); ++row)
        {
            output.at(row, col) = static_cast<float>(expValues[row * input.cols() + col] / sumExp);
        }
    }

    return output;
}

Tensor Softmax::forward(const Tensor& input) const
{
    if (!is_valid_tensor_matrix(input))
    {
        return Tensor{0, 0, 0.0f};
    }

    const SoftmaxConfig& config = this->config();

    if (config.axis == SoftmaxAxis::Row)
    {
        return forward_row(input, config);
    }

    return forward_col(input, config);
}

void Softmax::forward_inplace(Tensor& input) const
{
    Tensor temp = forward(input);
    input = temp;
}

const SoftmaxConfig& Softmax::config() const
{
    if (config_.temperature <= 0.0f)
    {
        throw std::invalid_argument("Softmax temperature must be greater than 0");
    }

    if (config_.epsilon < 0.0f)
    {
        throw std::invalid_argument("Softmax epsilon must be non-negative");
    }

    return config_;
}

Tensor softmax(const Tensor& input,
                 SoftmaxAxis axis,
                 float epsilon,
                 float temperature)
{
    return Softmax(SoftmaxConfig{axis, epsilon, temperature}).forward(input);
}
