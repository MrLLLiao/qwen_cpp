#include "model/model.h"
#include "model/model_config.h"
#include "model/model_weights.h"
#include "model/mlp.h"
#include "model/rms_norm.h"
#include "model/self-attention.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
constexpr float kEps = 1e-5f;

void expect_true(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "[FAIL] " << message << '\n';
        std::exit(1);
    }
}

bool nearly_equal(float a, float b)
{
    return std::fabs(a - b) <= kEps;
}

Tensor make_tensor(std::initializer_list<std::initializer_list<float>> rows)
{
    const size_t r = rows.size();
    const size_t c = rows.begin()->size();
    Tensor t(r, c, 0.0f);
    size_t i = 0;
    for (const auto& row : rows)
    {
        size_t j = 0;
        for (const float v : row)
        {
            t(i, j) = v;
            ++j;
        }
        ++i;
    }
    return t;
}

mini_llm::model::ModelConfig make_config()
{
    mini_llm::model::ModelConfig cfg;
    cfg.vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.hidden_size = 4;
    cfg.intermediate_size = 8;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.max_position_embeddings = 16;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 1000000.0f;
    cfg.rope_scale = 1.0f;
    return cfg;
}
} // namespace

int main()
{
    using namespace mini_llm::model;

    {
        const ModelConfig cfg = make_config();
        expect_true(cfg.valid(), "ModelConfig should validate");
        expect_true(cfg.head_dim() == 2, "head_dim should be hidden_size / num_attention_heads");
        expect_true(cfg.kv_hidden_size() == 2, "kv_hidden_size should use kv head count");

        ModelConfig invalid_hidden = cfg;
        invalid_hidden.hidden_size = 5;
        expect_true(!invalid_hidden.valid(), "ModelConfig should reject non-divisible hidden size");

        ModelConfig invalid_gqa = cfg;
        invalid_gqa.num_key_value_heads = 3;
        expect_true(!invalid_gqa.valid(), "ModelConfig should reject incompatible GQA heads");
    }

    {
        RMSNorm norm;
        const Tensor x = make_tensor({{1.0f, 2.0f, 3.0f, 4.0f}});
        const Tensor w = make_tensor({{1.0f, 1.0f, 1.0f, 1.0f}});
        const Tensor y = norm.forward(x, w);
        expect_true(y.rows() == 1 && y.cols() == 4, "RMSNorm output shape mismatch");
        expect_true(y(0, 0) < y(0, 3), "RMSNorm should preserve relative scale");
    }

    {
        MLP mlp(MLPConfig{4, 8});
        mlp.set_weights(Tensor(4, 8, 0.0f),
                        Tensor(4, 8, 0.0f),
                        Tensor(8, 4, 0.0f));
        const Tensor x = make_tensor({{1.0f, 2.0f, 3.0f, 4.0f}});
        const Tensor y = mlp.forward(x);
        expect_true(y.rows() == 1 && y.cols() == 4, "MLP output shape mismatch");
        expect_true(nearly_equal(y.max_value(), 0.0f), "zero weights should produce zero output");
    }

    {
        SelfAttention sa(SelfAttentionConfig{4, 2, 1, true, 0, 1000000.0f, 1.0f});
        sa.set_projection_weights(Tensor(4, 4, 0.0f),
                                  Tensor(4, 2, 0.0f),
                                  Tensor(4, 2, 0.0f),
                                  Tensor(4, 4, 0.0f));
        const Tensor x = make_tensor({{1.0f, 0.0f, 0.0f, 1.0f}});
        const Tensor y = sa.forward(x);
        expect_true(y.rows() == 1 && y.cols() == 4, "SelfAttention output shape mismatch");
    }

    {
        ModelWeights weights;
        weights.reset(make_config());
        weights.token_embedding() = Tensor(4, 4, 0.0f);
        weights.output_embedding() = Tensor(4, 4, 0.0f);

        ModelWeights::LayerWeights layer{};
        layer.attention_wq = Tensor(4, 4, 0.0f);
        layer.attention_wk = Tensor(4, 2, 0.0f);
        layer.attention_wv = Tensor(4, 2, 0.0f);
        layer.attention_wo = Tensor(4, 4, 0.0f);
        layer.mlp_gate = Tensor(4, 8, 0.0f);
        layer.mlp_up = Tensor(4, 8, 0.0f);
        layer.mlp_down = Tensor(8, 4, 0.0f);
        layer.rms_attn_weight = Tensor(1, 4, 1.0f);
        layer.rms_ffn_weight = Tensor(1, 4, 1.0f);
        weights.set_layer_weights(0, layer);

        expect_true(weights.ready(), "Weights should be ready after shape-filled reset");

        QwenModel model(std::move(weights));
        expect_true(model.ready(), "QwenModel should be ready after load");

        const Tensor embeddings = make_tensor({{1.0f, 0.0f, 0.0f, 1.0f}});
        const Tensor logits = model.forward(embeddings);
        expect_true(logits.rows() == 1 && logits.cols() == 4, "QwenModel logits shape mismatch");
    }

    std::cout << "[PASS] model tests passed\n";
    return 0;
}
