#include <rmm/mr/device/device_memory_resource.hpp>

#include <gmock/gmock.h>

namespace rmm::test {

class mock_resource : public rmm::mr::device_memory_resource {
 public:
  MOCK_METHOD(bool, supports_streams, (), (const, override, noexcept));
  MOCK_METHOD(bool, supports_get_mem_info, (), (const, override, noexcept));
  MOCK_METHOD(void*, do_allocate, (std::size_t, cuda_stream_view), (override));
  MOCK_METHOD(void, do_deallocate, (void*, std::size_t, cuda_stream_view), (override));
  using size_pair = std::pair<std::size_t, std::size_t>;
  MOCK_METHOD(size_pair, do_get_mem_info, (cuda_stream_view), (const, override));
};

}  // namespace rmm::test
