#include "cache/KVCache.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
constexpr float kEps = 1e-6F;

bool nearly_equal(const float a, const float b)
{
    return std::fabs(a - b) < kEps;
}

void expect_true(const bool condition, const char* message)
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
    Tensor2D t(r, c, 0.0F);

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

void test_reset_and_basic_state()
{
    KVCache cache;
    expect_true(!cache.initialized(), "default KVCache should be uninitialized");

    KVCache::Config cfg;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.head_dim = 2;
    cfg.max_tokens = 8;

    cache.reset(cfg);
    expect_true(cache.initialized(), "cache should be initialized after reset(config)");
    expect_true(cache.has_layer(0), "layer 0 should exist");
    expect_true(cache.has_layer(1), "layer 1 should exist");
    expect_true(!cache.has_layer(2), "layer 2 should not exist");

    expect_true(cache.token_count(0) == 0, "initial token_count(layer0) should be 0");
    expect_true(cache.token_count(1) == 0, "initial token_count(layer1) should be 0");
    expect_true(cache.total_token_count() == 0, "initial total_token_count should be 0");

    expect_true(cache.key(0).rows() == 0 && cache.key(0).cols() == 0, "initial key tensor should be empty");
    expect_true(cache.value(0).rows() == 0 && cache.value(0).cols() == 0, "initial value tensor should be empty");

    cache.reset();
    expect_true(!cache.initialized(), "cache should be uninitialized after reset()");
}

void test_append_and_growth()
{
    KVCache cache;
    KVCache::Config cfg{1, 2, 2, 6}; // cols should be num_heads * head_dim = 4
    cache.reset(cfg);

    const Tensor2D key1 = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F},
        {5.0F, 6.0F, 7.0F, 8.0F}
    });
    const Tensor2D val1 = make_tensor({
        {10.0F, 20.0F, 30.0F, 40.0F},
        {50.0F, 60.0F, 70.0F, 80.0F}
    });

    cache.append(0, key1, val1);

    expect_true(cache.token_count(0) == 2, "after first append token_count should be 2");
    expect_true(cache.key(0).rows() == 2 && cache.key(0).cols() == 4, "key shape after first append mismatch");
    expect_true(cache.value(0).rows() == 2 && cache.value(0).cols() == 4, "value shape after first append mismatch");
    expect_true(nearly_equal(cache.key(0)(1, 2), 7.0F), "key content after first append mismatch");
    expect_true(nearly_equal(cache.value(0)(0, 3), 40.0F), "value content after first append mismatch");

    const Tensor2D key2 = make_tensor({
        {9.0F, 10.0F, 11.0F, 12.0F}
    });
    const Tensor2D val2 = make_tensor({
        {90.0F, 100.0F, 110.0F, 120.0F}
    });

    cache.append(0, key2, val2);

    expect_true(cache.token_count(0) == 3, "after second append token_count should be 3");
    expect_true(cache.total_token_count() == 3, "total_token_count should be 3 for single-layer cache");

    // 验证前两行保留，第三行为新追加数据
    expect_true(nearly_equal(cache.key(0)(0, 0), 1.0F), "row0 should keep old key data");
    expect_true(nearly_equal(cache.key(0)(1, 3), 8.0F), "row1 should keep old key data");
    expect_true(nearly_equal(cache.key(0)(2, 1), 10.0F), "row2 should be appended key data");

    expect_true(nearly_equal(cache.value(0)(0, 2), 30.0F), "row0 should keep old value data");
    expect_true(nearly_equal(cache.value(0)(1, 0), 50.0F), "row1 should keep old value data");
    expect_true(nearly_equal(cache.value(0)(2, 3), 120.0F), "row2 should be appended value data");
}

void test_append_validation()
{
    KVCache cache;
    cache.reset(KVCache::Config{1, 2, 2, 2});

    const Tensor2D good = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F},
        {5.0F, 6.0F, 7.0F, 8.0F}
    });

    bool thrown = false;
    try
    {
        cache.append(1, good, good);
    }
    catch (const std::out_of_range&)
    {
        thrown = true;
    }
    expect_true(thrown, "append should throw for invalid layer index");

    thrown = false;
    try
    {
        const Tensor2D bad_value = make_tensor({
            {1.0F, 2.0F, 3.0F, 4.0F}
        });
        cache.append(0, good, bad_value);
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "append should throw for key/value shape mismatch");

    thrown = false;
    try
    {
        const Tensor2D bad_cols = make_tensor({
            {1.0F, 2.0F, 3.0F},
            {4.0F, 5.0F, 6.0F}
        });
        cache.append(0, bad_cols, bad_cols);
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "append should throw when cols != num_heads * head_dim");

    cache.append(0, good, good); // 用满 max_tokens = 2

    thrown = false;
    try
    {
        const Tensor2D one_more = make_tensor({
            {9.0F, 10.0F, 11.0F, 12.0F}
        });
        cache.append(0, one_more, one_more);
    }
    catch (const std::runtime_error&)
    {
        thrown = true;
    }
    expect_true(thrown, "append should throw when exceeding max_tokens");
}

void test_views()
{
    KVCache cache;
    cache.reset(KVCache::Config{1, 2, 2, 8});

    const Tensor2D key = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F},
        {5.0F, 6.0F, 7.0F, 8.0F},
        {9.0F, 10.0F, 11.0F, 12.0F}
    });
    const Tensor2D val = make_tensor({
        {10.0F, 20.0F, 30.0F, 40.0F},
        {50.0F, 60.0F, 70.0F, 80.0F},
        {90.0F, 100.0F, 110.0F, 120.0F}
    });
    cache.append(0, key, val);

    const KVCache::Tensor2DView kv = cache.key_view(0, 1, KVCache::kAllRows);
    expect_true(!kv.empty(), "key_view should not be empty");
    expect_true(kv.rows() == 2 && kv.cols() == 4, "key_view shape mismatch");
    expect_true(nearly_equal(kv(0, 0), 5.0F), "key_view first element mismatch");
    expect_true(nearly_equal(kv(1, 3), 12.0F), "key_view last element mismatch");

    const KVCache::Tensor2DView vv = cache.value_view(0, 2, 1);
    expect_true(vv.rows() == 1 && vv.cols() == 4, "value_view shape mismatch");
    expect_true(nearly_equal(vv(0, 2), 110.0F), "value_view element mismatch");

    const KVCache::Tensor2DView empty_tail = cache.key_view(0, 3, KVCache::kAllRows);
    expect_true(empty_tail.empty(), "view at row_offset==rows should be empty");

    bool thrown = false;
    try
    {
        static_cast<void>(cache.key_view(0, 4, 1));
    }
    catch (const std::out_of_range&)
    {
        thrown = true;
    }
    expect_true(thrown, "key_view should throw when row_offset > rows");
}
} // namespace

int main()
{
    test_reset_and_basic_state();
    test_append_and_growth();
    test_append_validation();
    test_views();

    std::cout << "[PASS] KVCache tests passed.\n";
    return 0;
}
