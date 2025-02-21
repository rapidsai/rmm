/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {
/**
 * @brief Converts a tuple into a parameter pack.
 *
 * This helper function for make_resource allows passing the upstreams as a
 * list of arguments to the Resource's constructor.
 *
 * @tparam Resource The resource type to create
 * @tparam UpstreamTuple A tuple of shared pointers of the types of the upstream resources
 * @tparam Args The types of the arguments to the resource's constructor
 * @param upstreams Tuple of `std::shared_ptr`s to the upstreams used by the wrapped resource, in
 * the same order as expected by `Resource`s constructor.
 * @param args Function parameter pack of arguments to forward to the Resource's
 * constructor
 * @return std::unique_ptr<Resource> A unique pointer to the created resource.
 */
template <typename Resource, typename UpstreamTuple, std::size_t... Indices, typename... Args>
auto make_resource_impl(UpstreamTuple const& upstreams,
                        std::index_sequence<Indices...>,
                        Args&&... args)
{
  return std::make_unique<Resource>(std::get<Indices>(upstreams).get()...,
                                    std::forward<Args>(args)...);
}

/**
 * @brief Create a `std::unique_ptr` to a `Resource` with the given upstreams and arguments
 *
 * @tparam Resource The resource type to create
 * @tparam Upstreams The types of the upstream resources
 * @tparam Args The types of the arguments to the resource's constructor
 * @param upstreams Tuple of `std::shared_ptr`s to the upstreams used by the wrapped resource, in
 * the same order as expected by `Resource`s constructor.
 * @param args Function parameter pack of arguments to forward to the wrapped resource's
 * constructor
 * @return std::unique_ptr<Resource> A unique pointer to the created resource
 */
template <typename Resource, typename... Upstreams, typename... Args>
auto make_resource(std::tuple<std::shared_ptr<Upstreams>...> const& upstreams, Args&&... args)
{
  return make_resource_impl<Resource>(
    upstreams, std::index_sequence_for<Upstreams...>{}, std::forward<Args>(args)...);
}
}  // namespace detail

/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource adaptor that maintains the lifetime of upstream resources.
 *
 * Many `device_memory_resource` derived types allocate memory from another "upstream" resource.
 * E.g., `pool_memory_resource` allocates its pool from an upstream resource. Typically, a resource
 * does not own its upstream, and therefore it is the user's responsibility to maintain the lifetime
 * of the upstream resource. This can be inconvenient and error prone, especially for resources with
 * complex upstreams that may themselves also have an upstream.
 *
 * `owning_wrapper` simplifies lifetime management of a resource, `wrapped`, by taking shared
 * ownership of all upstream resources via a `std::shared_ptr`.
 *
 * For convenience, it is recommended to use the `make_owning_wrapper` factory instead of
 * constructing an `owning_wrapper` directly.
 *
 * Example:
 * \code{.cpp}
 * auto cuda = std::make_shared<rmm::mr::cuda_memory_resource>();
 * auto pool = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(cuda,initial_pool_size,
 *                                                                         max_pool_size);
 * // The `cuda` resource will be kept alive for the lifetime of `pool` and automatically be
 * // destroyed after `pool` is destroyed
 * \endcode
 *
 * @tparam Resource Type of the wrapped resource
 * @tparam Upstreams Template parameter pack of the types of the upstream resources used by
 * `Resource`
 */
template <typename Resource, typename... Upstreams>
class owning_wrapper : public device_memory_resource {
 public:
  using upstream_tuple =
    std::tuple<std::shared_ptr<Upstreams>...>;  ///< Tuple of upstream memory resources

  /**
   * @brief Constructs the wrapped resource using the provided upstreams and any additional
   * arguments forwarded to the wrapped resources constructor.
   *
   * `Resource` is required to have a constructor whose first argument(s) are raw pointers to its
   * upstream resources in the same order as `upstreams`, followed by any additional arguments in
   * the same order as `args`.
   *
   * Example:
   * \code{.cpp}
   * template <typename Upstream1, typename Upstream2>
   * class example_resource{
   *   example_resource(Upstream1 * u1, Upstream2 * u2, int n, float f);
   * };
   *
   * using cuda = rmm::mr::cuda_memory_resource;
   * using example = example_resource<cuda,cuda>;
   * using wrapped_example = rmm::mr::owning_wrapper<example, cuda, cuda>;
   * auto cuda_mr = std::make_shared<cuda>();
   *
   * // Constructs an `example_resource` wrapped by an `owning_wrapper` taking shared ownership of
   * //`cuda_mr` and using it as both of `example_resource`s upstream resources. Forwards the
   * // arguments `42` and `3.14` to the additional `n` and `f` arguments of `example_resources`
   * // constructor.
   * wrapped_example w{std::make_tuple(cuda_mr,cuda_mr), 42, 3.14};
   * \endcode
   *
   * @tparam Args Template parameter pack to forward to the wrapped resource's constructor
   * @param upstreams Tuple of `std::shared_ptr`s to the upstreams used by the wrapped resource, in
   * the same order as expected by `Resource`s constructor.
   * @param args Function parameter pack of arguments to forward to the wrapped resource's
   * constructor
   */
  template <typename... Args>
  owning_wrapper(upstream_tuple upstreams, Args&&... args)
    : upstreams_{std::move(upstreams)},
      wrapped_{detail::make_resource<Resource>(upstreams_, std::forward<Args>(args)...)}
  {
  }

  /**
   * @briefreturn{A constant reference to the wrapped resource}
   */
  [[nodiscard]] Resource const& wrapped() const noexcept { return *wrapped_; }

  /**
   * @briefreturn{A reference to the wrapped resource}
   */
  [[nodiscard]] Resource& wrapped() noexcept { return *wrapped_; }

 private:
  /**
   * @brief Allocates memory using the wrapped resource.
   *
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled by the wrapped
   * resource.
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return void* Pointer to the memory allocated by the wrapped resource
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    return wrapped().allocate(bytes, stream);
  }

  /**
   * @brief Returns an allocation to the wrapped resource.
   *
   * `ptr` must have been returned from a prior call to `do_allocate(bytes)`.
   *
   * @param ptr Pointer to the allocation to free.
   * @param bytes Size of the allocation
   * @param stream Stream on which to deallocate the memory
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    wrapped().deallocate(ptr, bytes, stream);
  }

  /**
   * @brief Compare if this resource is equal to another.
   *
   * Two resources are equal if memory allocated by one resource can be freed by the other.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equal
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto casted = dynamic_cast<owning_wrapper<Resource, Upstreams...> const*>(&other);
    if (nullptr != casted) { return wrapped().is_equal(casted->wrapped()); }
    return wrapped().is_equal(other);
  }

  upstream_tuple upstreams_;           ///< The owned upstream resources
  std::unique_ptr<Resource> wrapped_;  ///< The wrapped resource that uses the upstreams
};

/**
 * @brief Constructs a resource of type `Resource` wrapped in an `owning_wrapper` using `upstreams`
 * as the upstream resources and `args` as the additional parameters for the constructor of
 * `Resource`.
 *
 * \code{.cpp}
 * template <typename Upstream1, typename Upstream2>
 * class example_resource{
 *   example_resource(Upstream1 * u1, Upstream2 * u2, int n, float f);
 * };
 *
 * auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
 * auto cuda_upstreams = std::make_tuple(cuda_mr, cuda_mr);
 *
 * // Constructs an `example_resource<rmm::mr::cuda_memory_resource, rmm::mr::cuda_memory_resource>`
 * // wrapped by an `owning_wrapper` taking shared ownership of `cuda_mr` and using it as both of
 * // `example_resource`s upstream resources. Forwards the  arguments `42` and `3.14` to the
 * // additional `n` and `f` arguments of `example_resource` constructor.
 * auto wrapped_example = rmm::mr::make_owning_wrapper<example_resource>(cuda_upstreams, 42, 3.14);
 * \endcode
 *
 * @tparam Resource Template template parameter specifying the type of the wrapped resource to
 * construct
 * @tparam Upstreams Types of the upstream resources
 * @tparam Args Types of the arguments used in `Resource`s constructor
 * @param upstreams Tuple of `std::shared_ptr`s to the upstreams used by the wrapped resource, in
 * the same order as expected by `Resource`s constructor.
 * @param args Function parameter pack of arguments to forward to the wrapped resource's
 * constructor
 * @return An `owning_wrapper` wrapping a newly constructed `Resource<Upstreams...>` and
 * `upstreams`.
 */
template <template <typename...> class Resource, typename... Upstreams, typename... Args>
auto make_owning_wrapper(std::tuple<std::shared_ptr<Upstreams>...> upstreams, Args&&... args)
{
  return std::make_shared<owning_wrapper<Resource<Upstreams...>, Upstreams...>>(
    std::move(upstreams), std::forward<Args>(args)...);
}

/**
 * @brief Additional convenience factory for `owning_wrapper` when `Resource` has only a single
 * upstream resource.
 *
 * When a resource has only a single upstream, it can be inconvenient to construct a `std::tuple` of
 * the upstream resource. This factory allows specifying the single upstream as just a
 * `std::shared_ptr`.
 *
 * @tparam Resource Type of the wrapped resource to construct
 * @tparam Upstream Type of the single upstream resource
 * @tparam Args Types of the arguments used in `Resource`s constructor
 * @param upstream `std::shared_ptr` to the upstream resource
 * @param args Function parameter pack of arguments to forward to the wrapped resource's constructor
 * @return An `owning_wrapper` wrapping a newly construct `Resource<Upstream>` and `upstream`.
 */
template <template <typename> class Resource, typename Upstream, typename... Args>
auto make_owning_wrapper(std::shared_ptr<Upstream> upstream, Args&&... args)
{
  return make_owning_wrapper<Resource>(std::make_tuple(std::move(upstream)),
                                       std::forward<Args>(args)...);
}

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
