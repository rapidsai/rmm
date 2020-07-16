/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "device_memory_resource.hpp"

#include <iostream>
#include <memory>
#include <utility>

namespace rmm {
namespace mr {

template <typename Resource, typename UpstreamTuple, std::size_t... Indices, typename... Args>
auto make_resource_impl(UpstreamTuple t, std::index_sequence<Indices...>, Args&&... args)
{
  return Resource{std::get<Indices>(t).get()..., std::forward<Args>(args)...};
}

template <typename Resource, typename... Upstreams, typename... Args>
auto make_resource(std::tuple<std::shared_ptr<Upstreams>...> t, Args&&... args)
{
  return make_resource_impl<Resource>(
    std::move(t), std::index_sequence_for<Upstreams...>{}, std::forward<Args>(args)...);
}

template <typename Resource, typename... Upstreams>
class owning_wrapper final : public device_memory_resource {
 public:
  using upstream_tuple = std::tuple<std::shared_ptr<Upstreams>...>;

  template <typename... Args>
  owning_wrapper(upstream_tuple upstreams, Args&&... args)
    : upstreams_{std::move(upstreams)},
      wrapped_{make_resource<Resource>(std::move(upstreams), std::forward<Args>(args)...)}
  {
    std::cout << "owning_wrapper. Number of args: " << sizeof...(args)
              << " Number of upstreams: " << std::tuple_size<upstream_tuple>::value << std::endl;
  }

  Resource const& wrapped() const noexcept { return wrapped_; }

  Resource& wrapped() noexcept { return wrapped_; }

 private:
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    return wrapped_.do_allocate(bytes, stream);
  }

  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    wrapped_.do_deallocate(p, bytes, stream);
  }

  bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) {
      return true;
    } else {
      auto casted = dynamic_cast<owning_wrapper<Resource, Upstreams...> const*>(&other);
      if (nullptr != casted) {
        return wrapped().is_equal(casted->wrapped());
      } else {
        return wrapped_.is_equal(other);
      }
    }
  }

  upstream_tuple upstreams_;
  Resource wrapped_;
};

template <template <typename...> class Resource, typename... Upstreams, typename... Args>
auto make_owning_wrapper(std::tuple<std::shared_ptr<Upstreams>...> upstreams, Args&&... args)
{
  return std::make_shared<owning_wrapper<Resource<Upstreams...>, Upstreams...>>(
    std::move(upstreams), std::forward<Args>(args)...);
}

}  // namespace mr
}  // namespace rmm
