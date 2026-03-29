#include "cache/CacheAllocator.h"

#include <stdexcept>

CacheAllocator::CacheAllocator() = default;

CacheAllocator::CacheAllocator(size_t max_buffers)
    : max_buffers_(max_buffers)
{
}

void CacheAllocator::configure(size_t max_buffers)
{
    // TODO: 接入真实内存池/页管理策略并做参数校验。
    max_buffers_ = max_buffers;
    if (used_buffers_ > max_buffers_)
    {
        used_buffers_ = max_buffers_;
    }
}

void CacheAllocator::reset()
{
    // TODO: 回收所有已分配缓存块。
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
    (void)spec;
    // TODO: 按 AllocationSpec 从内存池分配并跟踪生命周期。
    throw std::logic_error("TODO: CacheAllocator::allocate is not implemented");
}

std::vector<Tensor2D> CacheAllocator::allocate_batch(const AllocationSpec& spec)
{
    (void)spec;
    // TODO: 批量分配缓存块，优化碎片与分配开销。
    throw std::logic_error("TODO: CacheAllocator::allocate_batch is not implemented");
}

void CacheAllocator::release(const Tensor2D& buffer)
{
    (void)buffer;
    // TODO: 根据 buffer 元信息归还到内存池，并更新 used_buffers_。
    throw std::logic_error("TODO: CacheAllocator::release is not implemented");
}
