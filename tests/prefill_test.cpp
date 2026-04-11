#include "engine/prefill.h"

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
    config.kv_config = KVCache::Config{2, 2, 2, 4};
    config.max_active_caches = 2;
    config.allocator_max_buffers = 8;
    manager.configure(config);
    return manager;
}

void test_append_prefill_kv_basic_and_skip_null_layer()
{
    KVCache cache(KVCache::Config{2, 2, 2, 4});

    const Tensor2D key = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F},
        {5.0F, 6.0F, 7.0F, 8.0F}
    });
    const Tensor2D value = make_tensor({
        {10.0F, 20.0F, 30.0F, 40.0F},
        {50.0F, 60.0F, 70.0F, 80.0F}
    });

    const std::vector<PrefillLayerKV> req = {
        PrefillLayerKV{0, &key, &value},
        PrefillLayerKV{1, nullptr, nullptr}
    };

    append_prefill_kv(cache, req);

    expect_true(cache.token_count(0) == 2, "append_prefill_kv should append valid layer");
    expect_true(cache.token_count(1) == 0, "append_prefill_kv should skip null/null layer");
    expect_true(nearly_equal(cache.key(0)(1, 2), 7.0F), "appended key content mismatch");
}

void test_append_prefill_kv_invalid_pair_throws()
{
    KVCache cache(KVCache::Config{2, 2, 2, 4});
    const Tensor2D key = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });

    bool thrown = false;
    try
    {
        append_prefill_kv(cache, std::vector<PrefillLayerKV>{PrefillLayerKV{0, &key, nullptr}});
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "append_prefill_kv should throw when key/value are not paired");
}

void test_prefill_engine_run_success()
{
    CacheManager manager = make_configured_manager();
    manager.create_cache("session-a");

    const Tensor2D k0 = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F},
        {5.0F, 6.0F, 7.0F, 8.0F}
    });
    const Tensor2D v0 = make_tensor({
        {10.0F, 20.0F, 30.0F, 40.0F},
        {50.0F, 60.0F, 70.0F, 80.0F}
    });
    const Tensor2D k1 = make_tensor({
        {0.1F, 0.2F, 0.3F, 0.4F},
        {0.5F, 0.6F, 0.7F, 0.8F}
    });
    const Tensor2D v1 = make_tensor({
        {1.1F, 1.2F, 1.3F, 1.4F},
        {1.5F, 1.6F, 1.7F, 1.8F}
    });

    const PrefillRequest req{
        "session-a",
        std::vector<PrefillLayerKV>{
            PrefillLayerKV{0, &k0, &v0},
            PrefillLayerKV{1, &k1, &v1}
        }
    };

    PrefillEngine engine(manager);
    const PrefillResult result = engine.run(req);

    const KVCache& cache = manager.cache("session-a");
    expect_true(result.appended_tokens == 2, "run should report appended token count");
    expect_true(cache.token_count(0) == 2 && cache.token_count(1) == 2,
                "run should append tokens for all provided layers");
    expect_true(nearly_equal(cache.value(1)(0, 3), 1.4F), "run should write value tensor content");
}

void test_prefill_engine_run_empty_or_missing_cache()
{
    CacheManager manager = make_configured_manager();
    PrefillEngine engine(manager);

    const PrefillResult empty_req_result = engine.run(PrefillRequest{});
    expect_true(empty_req_result.appended_tokens == 0, "empty request should return zero appended tokens");

    const Tensor2D k = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });
    const Tensor2D v = make_tensor({
        {5.0F, 6.0F, 7.0F, 8.0F}
    });

    const PrefillRequest missing_cache_req{
        "not-exist",
        std::vector<PrefillLayerKV>{PrefillLayerKV{0, &k, &v}}
    };

    const PrefillResult missing_cache_result = engine.run(missing_cache_req);
    expect_true(missing_cache_result.appended_tokens == 0,
                "missing cache request should return zero appended tokens");
}

void test_prefill_engine_run_validation_errors()
{
    CacheManager manager = make_configured_manager();
    manager.create_cache("session-b");
    PrefillEngine engine(manager);

    const Tensor2D k = make_tensor({
        {1.0F, 2.0F, 3.0F, 4.0F}
    });
    const Tensor2D v = make_tensor({
        {5.0F, 6.0F, 7.0F, 8.0F}
    });

    bool thrown = false;
    try
    {
        const PrefillRequest duplicate_layer_req{
            "session-b",
            std::vector<PrefillLayerKV>{
                PrefillLayerKV{0, &k, &v},
                PrefillLayerKV{0, &k, &v}
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
        const Tensor2D k2 = make_tensor({
            {1.0F, 2.0F, 3.0F, 4.0F},
            {5.0F, 6.0F, 7.0F, 8.0F}
        });
        const Tensor2D v2 = make_tensor({
            {9.0F, 10.0F, 11.0F, 12.0F},
            {13.0F, 14.0F, 15.0F, 16.0F}
        });

        const PrefillRequest inconsistent_tokens_req{
            "session-b",
            std::vector<PrefillLayerKV>{
                PrefillLayerKV{0, &k, &v},
                PrefillLayerKV{1, &k2, &v2}
            }
        };
        static_cast<void>(engine.run(inconsistent_tokens_req));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "run should throw on inconsistent token count across layers");
}
} // namespace

int main()
{
    test_append_prefill_kv_basic_and_skip_null_layer();
    test_append_prefill_kv_invalid_pair_throws();
    test_prefill_engine_run_success();
    test_prefill_engine_run_empty_or_missing_cache();
    test_prefill_engine_run_validation_errors();

    std::cout << "[PASS] Prefill tests passed.\n";
    return 0;
}
