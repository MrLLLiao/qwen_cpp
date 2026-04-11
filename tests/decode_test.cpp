#include "engine/decode.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "cache/CacheManager.h"

namespace
{
constexpr float kEps = 1e-6F;

void expect_true(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "[FAIL] " << message << '\n';
        std::exit(1);
    }
}

bool nearly_equal(const float a, const float b)
{
    return std::fabs(a - b) < kEps;
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

CacheManager make_configured_manager()
{
    CacheManager manager;
    CacheManager::ManagerConfig config{};
    config.kv_config = KVCache::Config{2, 2, 2, 3};
    config.max_active_caches = 2;
    config.allocator_max_buffers = 8;
    manager.configure(config);
    return manager;
}

void test_append_decode_kv_basic_and_skip_null_layer()
{
    KVCache cache(KVCache::Config{2, 2, 2, 3});

    const Tensor2D key = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });
    const Tensor2D value = make_tensor({
        {10.0F, 20.0F, 30.0F, 40.0F}
    });

    const std::vector<DecodeLayerKV> req = {
        DecodeLayerKV{0, &key, &value},
        DecodeLayerKV{1, nullptr, nullptr}
    };

    append_decode_kv(cache, req);

    expect_true(cache.token_count(0) == 1, "append_decode_kv should append valid layer");
    expect_true(cache.token_count(1) == 0, "append_decode_kv should skip null/null layer");
    expect_true(nearly_equal(cache.key(0)(0, 2), 3.0F), "appended key content mismatch");
}

void test_append_decode_kv_invalid_pair_throws()
{
    KVCache cache(KVCache::Config{2, 2, 2, 3});
    const Tensor2D key = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });

    bool thrown = false;
    try
    {
        append_decode_kv(cache, std::vector<DecodeLayerKV>{DecodeLayerKV{0, &key, nullptr}});
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "append_decode_kv should throw when key/value are not paired");
}

void test_decode_engine_run_success()
{
    CacheManager manager = make_configured_manager();
    manager.create_cache("session-a");

    const Tensor2D k0 = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });
    const Tensor2D v0 = make_tensor({
        {10.0F, 20.0F, 30.0F, 40.0F}
    });
    const Tensor2D k1 = make_tensor({
        {0.1F, 0.2F, 0.3F, 0.4F}
    });
    const Tensor2D v1 = make_tensor({
        {1.1F, 1.2F, 1.3F, 1.4F}
    });

    const DecodeRequest req{
        "session-a",
        std::vector<DecodeLayerKV>{
            DecodeLayerKV{0, &k0, &v0},
            DecodeLayerKV{1, &k1, &v1}
        }
    };

    DecodeEngine engine(manager);
    const DecodeResult result = engine.run(req);

    const KVCache& cache = manager.cache("session-a");
    expect_true(result.appended_tokens == 1, "run should report decode appended token count");
    expect_true(result.total_tokens == 1, "run should report total token count after append");
    expect_true(cache.token_count(0) == 1 && cache.token_count(1) == 1,
                "run should append decode token for all provided layers");
    expect_true(nearly_equal(cache.value(1)(0, 3), 1.4F), "run should write value tensor content");
}

void test_decode_engine_run_empty_or_missing_cache()
{
    CacheManager manager = make_configured_manager();
    DecodeEngine engine(manager);

    const DecodeResult empty_req_result = engine.run(DecodeRequest{});
    expect_true(empty_req_result.appended_tokens == 0, "empty request should return zero appended tokens");
    expect_true(empty_req_result.total_tokens == 0, "empty request should return zero total tokens");

    const Tensor2D k = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });
    const Tensor2D v = make_tensor({
        {5.0F, 6.0F, 7.0F, 8.0F}
    });

    const DecodeRequest missing_cache_req{
        "not-exist",
        std::vector<DecodeLayerKV>{DecodeLayerKV{0, &k, &v}}
    };

    const DecodeResult missing_cache_result = engine.run(missing_cache_req);
    expect_true(missing_cache_result.appended_tokens == 0,
                "missing cache request should return zero appended tokens");
    expect_true(missing_cache_result.total_tokens == 0,
                "missing cache request should return zero total tokens");
}

void test_decode_engine_run_validation_errors()
{
    CacheManager manager = make_configured_manager();
    manager.create_cache("session-b");
    DecodeEngine engine(manager);

    const Tensor2D k = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });
    const Tensor2D v = make_tensor({
        {5.0F, 6.0F, 7.0F, 8.0F}
    });

    bool thrown = false;
    try
    {
        const DecodeRequest duplicate_layer_req{
            "session-b",
            std::vector<DecodeLayerKV>{
                DecodeLayerKV{0, &k, &v},
                DecodeLayerKV{0, &k, &v}
            }
        };
        static_cast<void>(engine.run(duplicate_layer_req));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "run should throw on duplicate layer index");

    thrown = false;
    try
    {
        const Tensor2D two_tokens = make_tensor({
            {1.0F, 2.0F, 3.0F, 4.0F},
            {5.0F, 6.0F, 7.0F, 8.0F}
        });

        const DecodeRequest multi_token_req{
            "session-b",
            std::vector<DecodeLayerKV>{
                DecodeLayerKV{0, &two_tokens, &two_tokens},
                DecodeLayerKV{1, &k, &v}
            }
        };
        static_cast<void>(engine.run(multi_token_req));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "run should throw when decode input has rows != 1");

    thrown = false;
    try
    {
        const Tensor2D bad_cols = make_tensor({
            {1.0F, 2.0F, 3.0F}
        });

        const DecodeRequest bad_hidden_req{
            "session-b",
            std::vector<DecodeLayerKV>{
                DecodeLayerKV{0, &bad_cols, &bad_cols},
                DecodeLayerKV{1, &k, &v}
            }
        };
        static_cast<void>(engine.run(bad_hidden_req));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "run should throw on invalid hidden size");

    const DecodeRequest ok_req{
        "session-b",
        std::vector<DecodeLayerKV>{
            DecodeLayerKV{0, &k, &v},
            DecodeLayerKV{1, &k, &v}
        }
    };
    static_cast<void>(engine.run(ok_req)); // total tokens = 1
    static_cast<void>(engine.run(ok_req)); // total tokens = 2
    static_cast<void>(engine.run(ok_req)); // total tokens = 3 == max

    thrown = false;
    try
    {
        static_cast<void>(engine.run(ok_req)); // exceed max_tokens
    }
    catch (const std::runtime_error&)
    {
        thrown = true;
    }
    expect_true(thrown, "run should throw when exceeding cache max_tokens");
}

void test_decode_engine_all_null_layers_returns_current_total_tokens()
{
    CacheManager manager = make_configured_manager();
    manager.create_cache("session-c");
    DecodeEngine engine(manager);

    const Tensor2D k = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });
    const Tensor2D v = make_tensor({
        {5.0F, 6.0F, 7.0F, 8.0F}
    });

    const DecodeRequest one_step_req{
        "session-c",
        std::vector<DecodeLayerKV>{
            DecodeLayerKV{0, &k, &v},
            DecodeLayerKV{1, &k, &v}
        }
    };
    static_cast<void>(engine.run(one_step_req));

    const DecodeRequest all_null_req{
        "session-c",
        std::vector<DecodeLayerKV>{
            DecodeLayerKV{0, nullptr, nullptr},
            DecodeLayerKV{1, nullptr, nullptr}
        }
    };

    const DecodeResult result = engine.run(all_null_req);
    expect_true(result.appended_tokens == 0, "all-null decode request should append zero tokens");
    expect_true(result.total_tokens == 1, "all-null decode request should report current total tokens");
}
} // namespace

int main()
{
    test_append_decode_kv_basic_and_skip_null_layer();
    test_append_decode_kv_invalid_pair_throws();
    test_decode_engine_run_success();
    test_decode_engine_run_empty_or_missing_cache();
    test_decode_engine_run_validation_errors();
    test_decode_engine_all_null_layers_returns_current_total_tokens();

    std::cout << "[PASS] Decode tests passed.\n";
    return 0;
}
