#pragma once

#include <cstddef>

// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {
namespace mr {
/**---------------------------------------------------------------------------*
 * @brief Base class for all libcudf device memory allocation.
 *
 * This class serves as the interface that all custom device memory
 * implementations must satisfy.
 *
 * There are two private, pure virtual functions: `do_allocate` and
 * `do_deallocate` that must be implemented. Optionally, the `is_equal` virtual
 * function may be overriden from the default of simply performing an identity
 * comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal`
 * simply call the private virtual functions. The reason for this is to allow
 * implementing shared, default behavior in the base class. For example, the
 * base class' `allocate` function may log every allocation, no matter what
 * derived class implementation is used.
 *
 *---------------------------------------------------------------------------**/
class device_memory_resource {
 public:
  device_memory_resource() = default;
  virtual ~device_memory_resource() = default;

  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @throws std::bad_alloc When the requested size cannot be allocated.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void* allocate(std::size_t bytes, cudaStream_t stream = 0) {
    return do_allocate(bytes, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes,stream)` on
   * a `device_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  void deallocate(void* p, std::size_t bytes, cudaStream_t stream = 0) {
    do_deallocate(p, bytes, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Compare this resource to another.
   *
   * Two device_memory_resources compare equal if and only if memory allocated
   * from one device_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @returns If the two resources are equivalent
   *---------------------------------------------------------------------------**/
  bool is_equal(device_memory_resource const& other) const noexcept {
    return do_is_equal(other);
  }

  /**---------------------------------------------------------------------------*
   * @brief Queries if the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns If the resource supports non-null streams
   *---------------------------------------------------------------------------**/
  virtual bool supports_streams() const noexcept = 0;

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  virtual void* do_allocate(std::size_t bytes, cudaStream_t stream) = 0;

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  virtual void do_deallocate(void* p, std::size_t bytes,
                             cudaStream_t stream) = 0;

  /**---------------------------------------------------------------------------*
   * @brief Compare this resource to another.
   *
   * Two device_memory_resources compare equal if and only if memory allocated
   * from one device_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   *---------------------------------------------------------------------------**/
  virtual bool do_is_equal(device_memory_resource const& other) const noexcept {
    return this == &other;
  }
};
}  // namespace mr
}  // namespace rmm