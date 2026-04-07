#include "cache/CacheManager.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
void expect_true(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "[FAIL] " << message << '\n';
        std::exit(1);
    }
}

KVCache::Config make_kv_config()
{
    return KVCache::Config{2, 2, 2, 8};
}

void test_config_validation_and_create_failure_paths()
{
    CacheManager manager;

    bool thrown = false;
    try
    {
        static_cast<void>(manager.create_cache("c0"));
    }
    catch (const std::logic_error&)
    {
        thrown = true;
    }
    expect_true(thrown, "create_cache should throw when manager is not configured");

    CacheManager::ManagerConfig bad = {KVCache::Config{0, 2, 2, 8}, 1, 4};
    thrown = false;
    try
    {
        manager.configure(bad);
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "configure should validate num_layers > 0");

    bad = {KVCache::Config{2, 0, 2, 8}, 1, 4};
    thrown = false;
    try
    {
        manager.configure(bad);
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "configure should validate num_heads > 0");

    bad = {KVCache::Config{2, 2, 0, 8}, 1, 4};
    thrown = false;
    try
    {
        manager.configure(bad);
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "configure should validate head_dim > 0");

    bad = {KVCache::Config{2, 2, 2, 0}, 1, 4};
    thrown = false;
    try
    {
        manager.configure(bad);
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "configure should validate max_tokens > 0");
}

void test_lifecycle_remove_and_clear_boundaries()
{
    CacheManager manager;
    manager.configure(CacheManager::ManagerConfig{make_kv_config(), 2, 8});

    KVCache& c0 = manager.create_cache("session-0");
    static_cast<void>(c0);
    expect_true(manager.active_cache_count() == 1, "active_cache_count should be 1 after first create");
    expect_true(manager.has_cache("session-0"), "has_cache should return true for created cache");

    manager.create_cache("session-1");
    expect_true(manager.active_cache_count() == 2, "active_cache_count should be 2 after second create");

    bool thrown = false;
    try
    {
        static_cast<void>(manager.create_cache("session-2"));
    }
    catch (const std::runtime_error&)
    {
        thrown = true;
    }
    expect_true(thrown, "create_cache should throw when max_active_caches reached");

    thrown = false;
    try
    {
        static_cast<void>(manager.create_cache("session-1"));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "create_cache should throw on duplicate cache id");

    manager.remove_cache("session-0");
    expect_true(!manager.has_cache("session-0"), "remove_cache should erase existing cache");
    expect_true(manager.active_cache_count() == 1, "active_cache_count should decrease after remove_cache");

    thrown = false;
    try
    {
        manager.remove_cache("missing");
    }
    catch (const std::out_of_range&)
    {
        thrown = true;
    }
    expect_true(thrown, "remove_cache should throw for missing cache id");

    manager.clear();
    expect_true(manager.active_cache_count() == 0, "clear should erase all caches");

    // clear on empty should be a no-op
    manager.clear();
    expect_true(manager.active_cache_count() == 0, "clear on empty manager should keep zero active caches");
}

void test_cache_accessors_and_allocator_reset()
{
    CacheManager manager;
    manager.configure(CacheManager::ManagerConfig{make_kv_config(), 2, 2});

    manager.create_cache("a");
    const CacheManager& const_mgr = manager;
    expect_true(const_mgr.cache("a").initialized(), "const cache accessor should return existing cache");

    bool thrown = false;
    try
    {
        static_cast<void>(manager.cache("missing"));
    }
    catch (const std::out_of_range&)
    {
        thrown = true;
    }
    expect_true(thrown, "cache(non-const) should throw for missing id");

    thrown = false;
    try
    {
        static_cast<void>(const_mgr.cache("missing"));
    }
    catch (const std::out_of_range&)
    {
        thrown = true;
    }
    expect_true(thrown, "cache(const) should throw for missing id");

    auto& allocator = manager.allocator();
    const Tensor2D temp = allocator.allocate(CacheAllocator::AllocationSpec{1, 1, 1});
    expect_true(allocator.used_buffers() == 1, "allocator should report used buffer after allocate");

    manager.clear();
    expect_true(manager.allocator().used_buffers() == 0, "clear should reset allocator usage");
    static_cast<void>(temp);
}
} // namespace

int main()
{
    test_config_validation_and_create_failure_paths();
    test_lifecycle_remove_and_clear_boundaries();
    test_cache_accessors_and_allocator_reset();

    std::cout << "[PASS] CacheManager tests passed.\n";
    return 0;
}
