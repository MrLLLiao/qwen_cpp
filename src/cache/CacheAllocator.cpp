#include "cache/CacheAllocator.h"

#include <stdexcept>

CacheAllocator::CacheAllocator() = default;

CacheAllocator::CacheAllocator(size_t max_buffers)
    : max_buffers_(max_buffers)
{
}

void CacheAllocator::configure(size_t max_buffers)
{
    max_buffers_ = max_buffers;
    used_buffers_ = 0;
    free_pool_.clear();
}

void CacheAllocator::reset()
{
    used_buffers_ = 0;
}

size_t CacheAllocator::max_buffers() const
{
    return max_buffers_;
}

size_t CacheAllocator::used_buffers() const
{
    return used_buffers_;
}

size_t CacheAllocator::free_buffers() const
{
    return max_buffers_ > used_buffers_ ? (max_buffers_ - used_buffers_) : 0;
}

Tensor2D CacheAllocator::allocate(const AllocationSpec& spec)
{
    if (spec.cols == 0 || spec.rows == 0)
    {
        throw std::invalid_argument("invalid shape");
    }
    if (used_buffers_ >= max_buffers_)
    {
        throw std::bad_alloc();
    }

    ShapeKey k{spec.rows, spec.cols};
    auto& bucket = free_pool_[k];

    Tensor2D out;
    if (!bucket.empty())
    {
        out = bucket.back();
        bucket.pop_back();
    }
    else
    {
        out = Tensor2D(spec.rows, spec.cols);
    }

    ++ used_buffers_;
    return out;
}

std::vector<Tensor2D> CacheAllocator::allocate_batch(const AllocationSpec& spec)
{
    if (spec.rows == 0 || spec.cols == 0)
    {
        throw std::invalid_argument("invalid shape");
    }

    if (spec.count == 0)
    {
        return {};
    }

    if (spec.count > free_buffers())
    {
        throw std::bad_alloc();
    }

    std::vector<Tensor2D> out;
    out.reserve(spec.count);

    for (size_t i = 0; i < spec.count; ++i)
    {
        out.push_back(allocate(spec));
    }

    return out;
}

void CacheAllocator::release(const Tensor2D& buffer)
{
    if (buffer.cols() == 0 || buffer.rows() == 0)
    {
        throw std::invalid_argument("invalid shape");
    }
    if (used_buffers_ == 0)
    {
        throw std::logic_error("CacheAllocator::release called when no buffers are in use");
    }

    ShapeKey k{buffer.rows(), buffer.cols()};
    auto& bucket = free_pool_[k];

    bucket.push_back(buffer);
    --used_buffers_;
}
