#include "cache/KVCache.h"
#include "ops/attention.h"
#include "tensor.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
struct BenchConfig
{
    std::size_t prompt_tokens{64};
    std::size_t decode_steps{32};
    std::size_t layers{2};
    std::size_t query_heads{8};
    std::size_t kv_heads{2};
    std::size_t head_dim{16};
};

std::size_t parse_size_arg(const char* text, const char* name)
{
    try
    {
        const std::size_t value = static_cast<std::size_t>(std::stoull(text));
        if (value == 0)
        {
            throw std::invalid_argument("zero");
        }
        return value;
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + text);
    }
}

BenchConfig parse_args(int argc, char** argv)
{
    BenchConfig config{};
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (i + 1 >= argc)
        {
            throw std::invalid_argument("missing value for " + arg);
        }

        if (arg == "--prompt")
        {
            config.prompt_tokens = parse_size_arg(argv[++i], "--prompt");
        }
        else if (arg == "--decode")
        {
            config.decode_steps = parse_size_arg(argv[++i], "--decode");
        }
        else if (arg == "--layers")
        {
            config.layers = parse_size_arg(argv[++i], "--layers");
        }
        else if (arg == "--q-heads")
        {
            config.query_heads = parse_size_arg(argv[++i], "--q-heads");
        }
        else if (arg == "--kv-heads")
        {
            config.kv_heads = parse_size_arg(argv[++i], "--kv-heads");
        }
        else if (arg == "--head-dim")
        {
            config.head_dim = parse_size_arg(argv[++i], "--head-dim");
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if ((config.query_heads % config.kv_heads) != 0)
    {
        throw std::invalid_argument("--q-heads must be divisible by --kv-heads");
    }
    return config;
}

float deterministic_value(std::size_t row, std::size_t col, float scale)
{
    const double x = static_cast<double>((row + 1U) * 17U + (col + 3U) * 31U);
    return static_cast<float>(std::sin(x * 0.013) * static_cast<double>(scale));
}

Tensor make_tensor(std::size_t rows, std::size_t cols, float scale, std::size_t row_bias = 0)
{
    Tensor tensor(rows, cols, 0.0f);
    for (std::size_t r = 0; r < rows; ++r)
    {
        float* row = tensor.row_data(r);
        for (std::size_t c = 0; c < cols; ++c)
        {
            row[c] = deterministic_value(r + row_bias, c, scale);
        }
    }
    return tensor;
}

void append_all_layers(KVCache& cache,
                       std::size_t layers,
                       const Tensor& key,
                       const Tensor& value)
{
    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        cache.append(layer, key, value);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const BenchConfig bench = parse_args(argc, argv);
        const std::size_t hidden_size = bench.query_heads * bench.head_dim;
        const std::size_t kv_width = bench.kv_heads * bench.head_dim;
        const std::size_t max_tokens = bench.prompt_tokens + bench.decode_steps;

        KVCache cache(KVCache::Config{
            bench.layers,
            bench.kv_heads,
            bench.head_dim,
            max_tokens,
            max_tokens,
        });

        const Tensor prompt_key = make_tensor(bench.prompt_tokens, kv_width, 0.02f);
        const Tensor prompt_value = make_tensor(bench.prompt_tokens, kv_width, 0.03f, 11U);
        append_all_layers(cache, bench.layers, prompt_key, prompt_value);

        AttentionConfig attention_config{};
        attention_config.causal = true;
        attention_config.rope_theta = 1000000.0f;
        attention_config.rope_scale = 1.0f;

        double checksum = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < bench.decode_steps; ++step)
        {
            Tensor hidden = make_tensor(1, hidden_size, 0.01f, bench.prompt_tokens + step);
            const Tensor next_key = make_tensor(1, kv_width, 0.02f, bench.prompt_tokens + step);
            const Tensor next_value = make_tensor(1, kv_width, 0.03f, bench.prompt_tokens + step + 11U);

            attention_config.query_position_offset = bench.prompt_tokens + step;
            for (std::size_t layer = 0; layer < bench.layers; ++layer)
            {
                cache.append(layer, next_key, next_value);
                hidden = grouped_query_attention(hidden,
                                                 cache.key(layer),
                                                 cache.value(layer),
                                                 bench.query_heads,
                                                 bench.kv_heads,
                                                 nullptr,
                                                 attention_config);
                checksum += static_cast<double>(hidden(0, layer % hidden.cols()));
            }
        }
        const auto stop = std::chrono::steady_clock::now();

        if (cache.total_token_count() != max_tokens)
        {
            throw std::runtime_error("KVCache token count mismatch after benchmark");
        }
        if (!std::isfinite(checksum))
        {
            throw std::runtime_error("benchmark checksum is not finite");
        }

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        const double decode_tokens_per_second =
            (static_cast<double>(bench.decode_steps) * 1000.0) / elapsed_ms;
        const double attention_calls_per_second =
            (static_cast<double>(bench.decode_steps * bench.layers) * 1000.0) / elapsed_ms;

        std::cout << "qwen_decode_bench\n"
                  << "  prompt_tokens: " << bench.prompt_tokens << '\n'
                  << "  decode_steps: " << bench.decode_steps << '\n'
                  << "  layers: " << bench.layers << '\n'
                  << "  query_heads: " << bench.query_heads << '\n'
                  << "  kv_heads: " << bench.kv_heads << '\n'
                  << "  head_dim: " << bench.head_dim << '\n'
                  << "  hidden_size: " << hidden_size << '\n'
                  << "  kv_width: " << kv_width << '\n'
                  << "  elapsed_ms: " << std::fixed << std::setprecision(3) << elapsed_ms << '\n'
                  << "  decode_tokens_per_second: " << std::setprecision(2)
                  << decode_tokens_per_second << '\n'
                  << "  attention_calls_per_second: " << attention_calls_per_second << '\n'
                  << "  kv_utilization_percent: " << cache.utilization(0) << '\n'
                  << "  checksum: " << std::setprecision(6) << checksum << '\n';
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[FAIL] " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
