#include "ops/attention.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
    bool nearly_equal(float a, float b, float eps = 1e-5f)
    {
        return std::fabs(a - b) <= eps;
    }

    void expect_true(bool condition, const char* message)
    {
        if (!condition)
        {
            std::cerr << "[FAIL] " << message << '\n';
            std::exit(1);
        }
    }

    Tensor2D make_tensor(std::initializer_list<std::initializer_list<float>> rows)
    {
        const size_t r = rows.size();
        const size_t c = rows.begin()->size();
        Tensor2D t(r, c, 0.0f);

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
}

int main()
{
    {
        const Tensor2D query = make_tensor({{1.0f, 0.0f}});
        const Tensor2D key = make_tensor({{1.0f, 0.0f}, {0.0f, 1.0f}});
        const Tensor2D value = make_tensor({{10.0f, 1.0f}, {0.0f, 5.0f}});

        AttentionConfig cfg;
        cfg.enable_scaling = false;

        const Tensor2D out = scaled_dot_product_attention(query, key, value, nullptr, cfg);

        expect_true(out.rows() == 1 && out.cols() == 2, "output shape should be 1x2");
        expect_true(nearly_equal(out(0, 0), 7.310586f), "basic attention out(0,0) mismatch");
        expect_true(nearly_equal(out(0, 1), 2.0757656f), "basic attention out(0,1) mismatch");
    }

    {
        const Tensor2D query = make_tensor({{1.0f, 0.0f}});
        const Tensor2D key = make_tensor({{1.0f, 0.0f}, {0.0f, 1.0f}});
        const Tensor2D value = make_tensor({{10.0f, 1.0f}, {0.0f, 5.0f}});
        const Tensor2D mask = make_tensor({{0.0f, -100.0f}});

        AttentionConfig cfg;
        cfg.enable_scaling = false;

        const Tensor2D out = scaled_dot_product_attention(query, key, value, &mask, cfg);

        expect_true(nearly_equal(out(0, 0), 10.0f, 1e-3f), "masked attention should focus first key for out(0,0)");
        expect_true(nearly_equal(out(0, 1), 1.0f, 1e-3f), "masked attention should focus first key for out(0,1)");
    }

    {
        const Tensor2D query = make_tensor({
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        });
        const Tensor2D& key = query;
        const Tensor2D value = make_tensor({
            {2.0f, 0.0f},
            {0.0f, 4.0f}
        });

        AttentionConfig cfg;
        cfg.enable_scaling = false;
        cfg.causal = true;

        const Tensor2D out = scaled_dot_product_attention(query, key, value, nullptr, cfg);

        expect_true(nearly_equal(out(0, 0), 2.0f, 1e-4f), "causal first token must only attend to itself (0,0)");
        expect_true(nearly_equal(out(0, 1), 0.0f, 1e-4f), "causal first token must only attend to itself (0,1)");
        expect_true(out(1, 0) > 0.0f && out(1, 0) < 2.0f, "causal second token should mix previous value for (1,0)");
        expect_true(out(1, 1) > 0.0f && out(1, 1) < 4.0f, "causal second token should mix current value for (1,1)");
    }

    {
        const Tensor2D query = make_tensor({{1.0f, 0.0f}});
        const Tensor2D key = make_tensor({{1.0f, 0.0f}, {0.0f, 1.0f}});
        const Tensor2D value = make_tensor({{10.0f, 1.0f}, {0.0f, 5.0f}});

        AttentionConfig cfg;
        cfg.enable_scaling = true;

        const Tensor2D out_scaled = scaled_dot_product_attention(query, key, value, nullptr, cfg);

        cfg.enable_scaling = false;
        const Tensor2D out_no_scale = scaled_dot_product_attention(query, key, value, nullptr, cfg);

        expect_true(out_scaled(0, 0) < out_no_scale(0, 0), "scaling should soften distribution on dominant value");
        expect_true(out_scaled(0, 1) > out_no_scale(0, 1), "scaling should increase weaker branch weight");
    }

    {
        bool thrown = false;
        try
        {
            const Tensor2D query(2, 3, 0.0f);
            const Tensor2D key(2, 4, 0.0f);
            const Tensor2D value(2, 5, 0.0f);
            static_cast<void>(scaled_dot_product_attention(query, key, value));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "attention should throw when query.cols() != key.cols()");

        thrown = false;
        try
        {
            const Tensor2D query(2, 3, 0.0f);
            const Tensor2D key(4, 3, 0.0f);
            const Tensor2D value(3, 2, 0.0f);
            static_cast<void>(scaled_dot_product_attention(query, key, value));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "attention should throw when key.rows() != value.rows()");

        thrown = false;
        try
        {
            const Tensor2D query(2, 3, 0.0f);
            const Tensor2D key(2, 3, 0.0f);
            const Tensor2D value(2, 2, 0.0f);
            const Tensor2D bad_mask(1, 2, 0.0f);
            static_cast<void>(scaled_dot_product_attention(query, key, value, &bad_mask));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "attention should throw on mask shape mismatch");

        thrown = false;
        try
        {
            const Tensor2D query(2, 2, 0.0f);
            const Tensor2D key(3, 2, 0.0f);
            const Tensor2D value(3, 2, 0.0f);
            AttentionConfig cfg;
            cfg.causal = true;
            static_cast<void>(scaled_dot_product_attention(query, key, value, nullptr, cfg));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "causal attention should require seq_q == seq_k");
    }

    std::cout << "[PASS] attention tests passed" << '\n';
    return 0;
}
