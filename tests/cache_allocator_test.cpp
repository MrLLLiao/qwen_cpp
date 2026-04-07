#include "cache/CacheAllocator.h"

#include <cstdlib>
#include <iostream>
#include <new>
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

void test_allocate_release_lifecycle()
{
    CacheAllocator allocator(2);
    CacheAllocator::AllocationSpec spec{2, 3, 1};

    const Tensor2D a = allocator.allocate(spec);
    expect_true(a.rows() == 2 && a.cols() == 3, "allocate should return requested shape");
    expect_true(allocator.used_buffers() == 1, "used_buffers should increase after allocate");
    expect_true(allocator.free_buffers() == 1, "free_buffers should decrease after allocate");

    const Tensor2D b = allocator.allocate(spec);
    expect_true(b.rows() == 2 && b.cols() == 3, "second allocate should return requested shape");
    expect_true(allocator.used_buffers() == 2, "used_buffers should be 2 after second allocate");

    bool thrown = false;
    try
    {
        static_cast<void>(allocator.allocate(spec));
    }
    catch (const std::bad_alloc&)
    {
        thrown = true;
    }
    expect_true(thrown, "allocate should throw bad_alloc when capacity exceeded");

    allocator.release(a);
    expect_true(allocator.used_buffers() == 1, "used_buffers should decrease after release");

    const Tensor2D c = allocator.allocate(spec);
    expect_true(c.rows() == 2 && c.cols() == 3, "re-allocate should still return requested shape");
    expect_true(allocator.used_buffers() == 2, "used_buffers should return to 2 after re-allocate");
}

void test_allocate_batch_and_reset()
{
    CacheAllocator allocator(4);
    CacheAllocator::AllocationSpec spec{3, 2, 3};

    const auto batch = allocator.allocate_batch(spec);
    expect_true(batch.size() == 3, "allocate_batch should return count elements");
    expect_true(allocator.used_buffers() == 3, "used_buffers should match batch count");

    allocator.reset();
    expect_true(allocator.used_buffers() == 0, "reset should clear used_buffers");
    expect_true(allocator.free_buffers() == 4, "reset should restore free_buffers");
}

void test_failure_paths()
{
    CacheAllocator allocator(1);

    bool thrown = false;
    try
    {
        static_cast<void>(allocator.allocate(CacheAllocator::AllocationSpec{0, 2, 1}));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "allocate should throw invalid_argument for zero rows");

    thrown = false;
    try
    {
        static_cast<void>(allocator.allocate(CacheAllocator::AllocationSpec{2, 0, 1}));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "allocate should throw invalid_argument for zero cols");

    thrown = false;
    try
    {
        static_cast<void>(allocator.allocate_batch(CacheAllocator::AllocationSpec{0, 2, 1}));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "allocate_batch should throw invalid_argument for invalid shape");

    const auto empty_batch = allocator.allocate_batch(CacheAllocator::AllocationSpec{2, 2, 0});
    expect_true(empty_batch.empty(), "allocate_batch with count=0 should return empty vector");

    thrown = false;
    try
    {
        static_cast<void>(allocator.allocate_batch(CacheAllocator::AllocationSpec{2, 2, 2}));
    }
    catch (const std::bad_alloc&)
    {
        thrown = true;
    }
    expect_true(thrown, "allocate_batch should throw bad_alloc when count > free_buffers");

    thrown = false;
    try
    {
        allocator.release(Tensor2D{});
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "release should throw invalid_argument for empty tensor");

    thrown = false;
    try
    {
        allocator.release(Tensor2D(1, 1, 0.0F));
    }
    catch (const std::logic_error&)
    {
        thrown = true;
    }
    expect_true(thrown, "release should throw logic_error when no active allocation exists");
}
} // namespace

int main()
{
    test_allocate_release_lifecycle();
    test_allocate_batch_and_reset();
    test_failure_paths();

    std::cout << "[PASS] CacheAllocator tests passed.\n";
    return 0;
}
