/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal reproducer for CCCL 3.2.x resource_ref conversion bug
// This reproduces the exact pattern from cudf's device_scalar that causes segfault

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/host_vector.h>

#include <gtest/gtest.h>

// Simplified rmm_host_allocator (like cudf's)
template <typename T>
class simple_host_allocator {
 public:
  using value_type = T;

  simple_host_allocator() = delete;

  // Constructor takes raw CCCL type (reproduces the bug)
  template <class... Properties>
  simple_host_allocator(cuda::mr::resource_ref<cuda::mr::host_accessible, Properties...> mr,
                        rmm::cuda_stream_view stream)
    : mr_(mr),  // Implicit conversion happens here!
      stream_(stream)
  {
    std::cout << "Allocator constructed (CCCL type), mr_=0x" << std::hex << (void*)&mr_ << std::dec
              << "\n"
              << std::flush;
  }

  // Alternative constructor that takes RMM wrapper directly (should work with conversion fix)
  template <typename OtherResourceType>
  simple_host_allocator(rmm::detail::cccl_async_resource_ref<OtherResourceType> mr,
                        rmm::cuda_stream_view stream)
    : mr_(mr),  // Uses conversion constructor!
      stream_(stream)
  {
    std::cout << "Allocator constructed (RMM wrapper), mr_=0x" << std::hex << (void*)&mr_
              << std::dec << "\n"
              << std::flush;
  }

  simple_host_allocator(simple_host_allocator const& other) : mr_(other.mr_), stream_(other.stream_)
  {
    std::cout << "Allocator copied from 0x" << std::hex << (void*)&other << " to 0x" << (void*)this
              << std::dec << "\n"
              << std::flush;
  }

  simple_host_allocator(simple_host_allocator&& other)
    : mr_(std::move(other.mr_)), stream_(other.stream_)
  {
    std::cout << "Allocator moved from 0x" << std::hex << (void*)&other << " to 0x" << (void*)this
              << std::dec << "\n"
              << std::flush;
  }

  ~simple_host_allocator()
  {
    std::cout << "~Allocator 0x" << std::hex << (void*)this << std::dec << "\n" << std::flush;
  }

  T* allocate(std::size_t n)
  {
    auto const result = mr_.allocate(stream_, n * sizeof(T));
    stream_.synchronize();
    return static_cast<T*>(result);
  }

  void deallocate(T* p, std::size_t n) noexcept
  {
    std::cout << "Allocator 0x" << std::hex << (void*)this << " deallocating, mr_=0x" << (void*)&mr_
              << std::dec << "\n"
              << std::flush;
    // This is where the segfault happens - mr_ has corrupt internal state
    mr_.deallocate(stream_, p, n * sizeof(T));
    std::cout << "Deallocate succeeded\n" << std::flush;
  }

  bool operator==(simple_host_allocator const& other) const { return mr_ == other.mr_; }
  bool operator!=(simple_host_allocator const& other) const { return !(*this == other); }

 private:
  rmm::host_async_resource_ref mr_;  // NOTE: Only host_accessible, not device_accessible!
  rmm::cuda_stream_view stream_;
};

// Function that returns host_device_async_resource_ref (like cudf::get_pinned_memory_resource)
rmm::host_device_async_resource_ref get_test_resource()
{
  static rmm::mr::pinned_host_memory_resource mr{};
  static rmm::host_device_async_resource_ref ref{mr};
  std::cout << "get_test_resource returning ref at 0x" << std::hex << (void*)&ref << std::dec
            << "\n"
            << std::flush;
  return ref;  // Returns copy of static ref
}

// Helper to disable RVO
template <typename T>
T&& disable_rvo(T& val)
{
  return std::move(val);
}

// Mimics cudf's make_pinned_vector_async pattern EXACTLY - inline allocator construction
template <typename T>
thrust::host_vector<T, simple_host_allocator<T>> make_test_vector_inline(
  std::size_t size, rmm::cuda_stream_view stream)
{
  // This matches cudf's pattern: pass resource inline to vector constructor
  return thrust::host_vector<T, simple_host_allocator<T>>(size, {get_test_resource(), stream});
}

// Mimics cudf's make_pinned_vector pattern - with named variable and synchronize
template <typename T>
thrust::host_vector<T, simple_host_allocator<T>> make_test_vector(std::size_t size,
                                                                  rmm::cuda_stream_view stream)
{
  std::cout << "Inside make_test_vector, before construction\n" << std::flush;
  // This matches cudf's pattern: create as named variable, then return it
  auto result = make_test_vector_inline<T>(size, stream);
  std::cout << "After vector construction, before synchronize\n" << std::flush;
  stream.synchronize();
  std::cout << "After synchronize, before return\n" << std::flush;
  return result;  // Returns named variable - may trigger move if NRVO doesn't apply
}

// Version that forces move (no RVO possible)
template <typename T>
thrust::host_vector<T, simple_host_allocator<T>> make_test_vector_no_rvo(
  std::size_t size, rmm::cuda_stream_view stream)
{
  auto result = thrust::host_vector<T, simple_host_allocator<T>>(
    size, simple_host_allocator<T>{get_test_resource(), stream});
  stream.synchronize();
  return disable_rvo(result);  // Forces move
}

// Test direct vector construction (no function call)
TEST(CCCL32ConversionBug, DirectVectorConstruction)
{
  rmm::cuda_stream stream{};

  std::cout << "Before vector construction\n" << std::flush;
  {
    // Create vector directly
    thrust::host_vector<int, simple_host_allocator<int>> vec(
      1, simple_host_allocator<int>{get_test_resource(), stream});
    std::cout << "After vector construction, before destructor\n" << std::flush;
  }

  // Destructor called here
  std::cout << "After vec destructor, test ending\n" << std::flush;
}

// Test explicit vector move
TEST(CCCL32ConversionBug, ExplicitVectorMove)
{
  rmm::cuda_stream stream{};

  // Create vector
  thrust::host_vector<int, simple_host_allocator<int>> vec1(
    1, simple_host_allocator<int>{get_test_resource(), stream});

  // Move it
  auto vec2 = std::move(vec1);

  // Destructor called on vec2
}

// Test move with nested scope (like function return)
TEST(CCCL32ConversionBug, NestedScopeMove)
{
  rmm::cuda_stream stream{};

  std::optional<thrust::host_vector<int, simple_host_allocator<int>>> vec_outer;

  {
    thrust::host_vector<int, simple_host_allocator<int>> vec_inner(
      1, simple_host_allocator<int>{get_test_resource(), stream});

    vec_outer = std::move(vec_inner);
    // vec_inner destroyed here
  }

  // vec_outer destroyed here
}

// Test inline allocator construction (like cudf's make_pinned_vector_async)
TEST(CCCL32ConversionBug, InlineAllocatorConstruction)
{
  rmm::cuda_stream stream{};

  std::cout << "Before make_test_vector_inline\n" << std::flush;
  {
    auto vec = make_test_vector_inline<int>(1, stream);
    std::cout << "After make_test_vector_inline, before destructor\n" << std::flush;
  }
  std::cout << "After vec destructor\n" << std::flush;
}

// Test just vector construction/destruction (through function)
TEST(CCCL32ConversionBug, JustVector)
{
  rmm::cuda_stream stream{};

  std::cout << "Before make_test_vector\n" << std::flush;
  {
    // Just create and destroy the vector
    auto vec = make_test_vector<int>(1, stream);
    std::cout << "After make_test_vector, before destructor\n" << std::flush;
  }
  // Destructor called here
  std::cout << "After vec destructor, before stream destructor\n" << std::flush;
}

// Test that reproduces the exact cudf pattern
TEST(CCCL32ConversionBug, DeviceScalarBounceBufferPattern)
{
  rmm::cuda_stream stream{};

  // Simulate device_scalar construction:
  // bounce_buffer{make_pinned_vector<T>(1, stream)}
  auto bounce_buffer = make_test_vector<int>(1, stream);

  // Use it
  bounce_buffer[0] = 42;
  ASSERT_EQ(bounce_buffer[0], 42);

  // Destructor will be called here - this should segfault with the bug
}

// Simpler direct conversion test
TEST(CCCL32ConversionBug, DirectConversion)
{
  rmm::cuda_stream stream{};

  // Get a resource with host+device properties
  auto mr_hostdevice = get_test_resource();

  // Create allocator (triggers conversion to host-only)
  simple_host_allocator<int> alloc{mr_hostdevice, stream};

  // Allocate
  int* ptr = alloc.allocate(10);
  ASSERT_NE(ptr, nullptr);

  // Deallocate - this should segfault with the bug
  alloc.deallocate(ptr, 10);
}

// Test allocator copy
TEST(CCCL32ConversionBug, AllocatorCopy)
{
  rmm::cuda_stream stream{};

  auto mr_hostdevice = get_test_resource();
  simple_host_allocator<int> alloc1{mr_hostdevice, stream};

  // Copy the allocator
  simple_host_allocator<int> alloc2 = alloc1;

  // Allocate with the copy
  int* ptr = alloc2.allocate(10);
  ASSERT_NE(ptr, nullptr);

  // Deallocate with the copy
  alloc2.deallocate(ptr, 10);
}
