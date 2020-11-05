// Copyright (c) 2020, NVIDIA CORPORATION.

#include <memory>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>

#include <thrust/optional.h>

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

class default_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  default_memory_resource_wrapper(int device)
    : mr(rmm::mr::get_per_device_resource(rmm::cuda_device_id(device)))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr()
  {
    return std::shared_ptr<rmm::mr::device_memory_resource>(mr, [](auto* p) {});
  }

 private:
  rmm::mr::device_memory_resource* mr;
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

class pool_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  pool_memory_resource_wrapper(
    std::shared_ptr<device_memory_resource_wrapper> upstream_mr,
    std::size_t initial_pool_size = ~0,  // TODO use std::optional / thrust::optional when available
    std::size_t maximum_pool_size = ~0)
    : upstream_mr(upstream_mr),
      mr(std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
        upstream_mr->get_mr().get(),
        initial_pool_size == static_cast<size_t>(~0) ? thrust::nullopt
                                                     : thrust::make_optional(initial_pool_size),
        maximum_pool_size == static_cast<size_t>(~0) ? thrust::nullopt
                                                     : thrust::make_optional(maximum_pool_size)))
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

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<device_memory_resource_wrapper> upstream_mr;
  std::shared_ptr<rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>> mr;
};

class binning_memory_resource_wrapper : public device_memory_resource_wrapper {
 public:
  binning_memory_resource_wrapper(std::shared_ptr<device_memory_resource_wrapper> upstream_mr)
    : upstream_mr(upstream_mr),
      mr(std::make_shared<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>>(
        upstream_mr->get_mr().get()))
  {
  }

  binning_memory_resource_wrapper(std::shared_ptr<device_memory_resource_wrapper> upstream_mr,
                                  int8_t min_size_exponent,
                                  int8_t max_size_exponent)
    : upstream_mr(upstream_mr),
      mr(std::make_shared<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>>(
        upstream_mr->get_mr().get(), min_size_exponent, max_size_exponent))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

  void add_bin(std::size_t allocation_size,
               std::shared_ptr<device_memory_resource_wrapper> bin_mr = {})
  {
    if (nullptr == bin_mr.get())
      mr->add_bin(allocation_size);
    else {
      bin_mrs.push_back(bin_mr);
      mr->add_bin(allocation_size, bin_mr->get_mr().get());
    }
  }

 private:
  std::shared_ptr<device_memory_resource_wrapper> upstream_mr;
  std::shared_ptr<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>> mr;
  std::vector<std::shared_ptr<device_memory_resource_wrapper>> bin_mrs;
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

class thread_safe_resource_adaptor_wrapper : public device_memory_resource_wrapper {
 public:
  thread_safe_resource_adaptor_wrapper(std::shared_ptr<device_memory_resource_wrapper> upstream_mr)
    : upstream_mr(upstream_mr),
      mr(std::make_shared<rmm::mr::thread_safe_resource_adaptor<rmm::mr::device_memory_resource>>(
        upstream_mr->get_mr().get()))
  {
  }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() { return mr; }

 private:
  std::shared_ptr<device_memory_resource_wrapper> upstream_mr;
  std::shared_ptr<rmm::mr::thread_safe_resource_adaptor<rmm::mr::device_memory_resource>> mr;
};

inline void set_per_device_resource(int device_id,
                                    std::shared_ptr<device_memory_resource_wrapper> new_resource)
{
  rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, new_resource->get_mr().get());
}

inline void set_current_device_resource(
  std::shared_ptr<device_memory_resource_wrapper> new_resource)
{
  rmm::mr::set_current_device_resource(new_resource->get_mr().get());
}
