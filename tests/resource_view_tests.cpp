#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

struct ResourceViewTest : public ::testing::Test {
};

void foo(rmm::mr::device_memory_resource* mr) {}

void bar(cuda::resource_view<cuda::memory_access::device> mr) {}

TEST_F(ResourceViewTest, InterchangeableWithResourcePointer)
{
  rmm::mr::cuda_memory_resource mr;
  EXPECT_EQ(cuda::stream_ordered_resource_view<cuda::memory_access::device>{&mr},
            cuda::view_resource(&mr));

  foo(&mr);
  bar(&mr);
  bar(cuda::view_resource(&mr));
}
