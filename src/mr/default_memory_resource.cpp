#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <atomic>

namespace rmm {
namespace mr {
namespace {
// The initial, default memory resource is a `cuda_memory_resource`
device_memory_resource* initial_resource() {
  static cuda_memory_resource resource{};
  return &resource;
}

std::atomic<device_memory_resource*>& get_default() {
  static std::atomic<device_memory_resource*> res{initial_resource()};
  return res;
}
}  // namespace

device_memory_resource* get_default_resource() { return get_default().load(); }

device_memory_resource* set_default_resource(
    device_memory_resource* new_resource) {
  if (nullptr == new_resource) {
    get_default().exchange(initial_resource());
  }

  return get_default().exchange(new_resource);
}
}  // namespace mr
}  // namespace rmm
