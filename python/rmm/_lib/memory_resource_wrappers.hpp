// Copyright (c) 2020, NVIDIA CORPORATION.

#include <memory>
#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_multisize_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/hybrid_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <string>
#include <vector>

// These are "owning" versions of the memory_resource classes
// that help lift the responsibility of managing memory resource
// lifetimes. For example, a `pool_memory_resource_wrapper`
// constructed using a `cuda_memory_resource_wrapper` as its
// "upstream" resource does not require the user to keep a reference
// to the latter.

class device_memory_resource_wrapper {
 public:
  virtual std::shared_ptr<rmm::mr::device_memory_resource> get_mr() = 0;
};

class cuda_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  cuda_memory_resource_wrapper() : mr(std::make_shared<rmm::mr::cuda_memory_resource>()) {}

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<rmm::mr::cuda_memory_resource> mr;
};

class managed_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  managed_memory_resource_wrapper() : mr(std::make_shared<rmm::mr::managed_memory_resource>()) {}

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<rmm::mr::managed_memory_resource> mr;
};

class cnmem_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  cnmem_memory_resource_wrapper(std::size_t initial_pool_size   = 0,
                                std::vector<int> const& devices = {})
    : mr(std::make_shared<rmm::mr::cnmem_memory_resource>(initial_pool_size, devices))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<rmm::mr::cnmem_memory_resource> mr;
};

class cnmem_managed_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  cnmem_managed_memory_resource_wrapper(std::size_t initial_pool_size   = 0,
                                        std::vector<int> const& devices = {})
    : mr(std::make_shared<rmm::mr::cnmem_managed_memory_resource>(initial_pool_size, devices))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<rmm::mr::cnmem_managed_memory_resource> mr;
};

class pool_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  pool_memory_resource_wrapper(std::shared_ptr<device_memory_resource_wrapper> upstream_mr,
                               std::size_t initial_pool_size,
                               std::size_t maximum_pool_size)
    : upstream_mr(upstream_mr),
      mr(std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
        upstream_mr->get_mr().get(), initial_pool_size, maximum_pool_size))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<device_memory_resource_wrapper> upstream_mr;
  std::shared_ptr<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> mr;
};

class fixed_size_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  fixed_size_memory_resource_wrapper(std::shared_ptr<device_memory_resource_wrapper> upstream_mr,
                                     std::size_t block_size,
                                     std::size_t blocks_to_preallocate)
    : upstream_mr(upstream_mr),
      mr(std::make_shared<rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>>(
        upstream_mr->get_mr().get(), block_size, blocks_to_preallocate))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource>

  get_mr()
  {
    return mr;
  }

 private:
  std::shared_ptr<device_memory_resource_wrapper> upstream_mr;
  std::shared_ptr<rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>> mr;
};

class fixed_multisize_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  fixed_multisize_memory_resource_wrapper(
    std::shared_ptr<device_memory_resource_wrapper> upstream_mr,
    std::size_t size_base,
    std::size_t min_size_exponent,
    std::size_t max_size_exponent,
    std::size_t initial_blocks_per_size)
    : upstream_mr(upstream_mr),
      mr(
        std::make_shared<rmm::mr::fixed_multisize_memory_resource<rmm::mr::device_memory_resource>>(
          upstream_mr->get_mr().get(),
          size_base,
          min_size_exponent,
          max_size_exponent,
          initial_blocks_per_size))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource>

  get_mr()
  {
    return mr;
  }

 private:
  std::shared_ptr<device_memory_resource_wrapper> upstream_mr;
  std::shared_ptr<rmm::mr::fixed_multisize_memory_resource<rmm::mr::device_memory_resource>> mr;
};

class hybrid_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  hybrid_memory_resource_wrapper(std::shared_ptr<device_memory_resource_wrapper> small_alloc_mr,
                                 std::shared_ptr<device_memory_resource_wrapper> large_alloc_mr,
                                 std::size_t threshold_size)
    : small_alloc_mr(small_alloc_mr),
      large_alloc_mr(large_alloc_mr),
      mr(std::make_shared<rmm::mr::hybrid_memory_resource<rmm::mr::device_memory_resource,
                                                          rmm::mr::device_memory_resource>>(
        small_alloc_mr->get_mr().get(), large_alloc_mr->get_mr().get(), threshold_size))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<device_memory_resource_wrapper> small_alloc_mr;
  std::shared_ptr<device_memory_resource_wrapper> large_alloc_mr;
  std::shared_ptr<rmm::mr::hybrid_memory_resource<rmm::mr::device_memory_resource,
                                                  rmm::mr::device_memory_resource>>
    mr;
};

class logging_resource_adaptor_wrapper : public device_memory_resource_wrapper {
 public:
  logging_resource_adaptor_wrapper(std::shared_ptr<device_memory_resource_wrapper> upstream_mr,
                                   const std::string& filename)
    : upstream_mr(upstream_mr),
      mr(std::make_shared<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>>(
        upstream_mr->get_mr().get(), filename))
  {
  }

  void flush() { mr->flush(); }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<device_memory_resource_wrapper> upstream_mr;
  std::shared_ptr<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>> mr;
};

void set_default_resource(std::shared_ptr<device_memory_resource_wrapper> new_resource)
{
  rmm::mr::set_default_resource(new_resource->get_mr().get());
}
