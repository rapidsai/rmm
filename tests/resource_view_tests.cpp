#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

struct ResourceViewTest : public ::testing::Test {
};

void foo(rmm::mr::device_memory_resource* mr) {}

void bar(cuda::pmr::resource_ptr<cuda::pmr::memory_access::device> mr) {}

TEST_F(ResourceViewTest, InterchangeableWithResourcePointer)
{
  rmm::mr::cuda_memory_resource mr;
  EXPECT_EQ(cuda::pmr::stream_ordered_resource_ptr<cuda::pmr::memory_access::device>{&mr}, &mr);

  foo(&mr);
  bar(&mr);
}

/*TEST_F(ResourceViewTest, KindFromProperties)
{
  static_assert(
    cuda::pmr::has_property<cuda::pmr::stream_ordered_resource_ptr<cuda::pmr::memory_kind::device>,
                            cuda::pmr::memory_access::device>::value,
    "Doesn't have property");
  static_assert(cuda::pmr::kind_has_property<cuda::pmr::memory_kind::device,
                                             cuda::pmr::memory_access::device>::value,
                "Doesn't have property");
  static_assert(cuda::pmr::kind_has_property<
                  cuda::pmr::detail::kind_from_properties<cuda::pmr::memory_access::device>,
                  cuda::pmr::memory_access::device>::value,
                "Doesn't have property");
  static_assert(cuda::pmr::has_property<
                  cuda::pmr::stream_ordered_resource<
                    cuda::pmr::detail::kind_from_properties<cuda::pmr::memory_access::device>>,
                  cuda::pmr::memory_access::device>::value,
                "Doesn't have property");
  cuda::pmr::stream_ordered_resource<
    cuda::pmr::detail::kind_from_properties<cuda::pmr::memory_access::device>>* mr{};
  cuda::pmr::stream_ordered_resource_ptr<cuda::pmr::memory_access::device> view{mr};
}*/
