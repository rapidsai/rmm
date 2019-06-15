#include "device_memory_resource.hpp"
namespace rmm {
namespace mr {

/**---------------------------------------------------------------------------*
 * @brief Get the default device memory resource pointer.
 *
 * The default device memory resource is used when an explicit memory resource
 * is not supplied. The initial default memory resource is a
 * `cuda_memory_resource`.
 *
 * The default memory resource is used for all temporary memory allocation.
 *
 * This function is thread-safe.
 *
 * @return device_memory_resource* Pointer to the current default memory
 * resource
 *---------------------------------------------------------------------------**/
device_memory_resource* get_default_resource();

/**---------------------------------------------------------------------------*
 * @brief Sets the default device memory resource pointer.
 *
 * If `new_resource` is not `nullptr`, sets the default device memory resource
 * pointer to `new_resource`. Otherwise, resets the default device memory
 * resource to the initial `cuda_memory_resource`.
 *
 * It is the caller's responsibility to maintain the lifetime of the object
 * pointed to by `new_resource`.
 *
 * This function is thread-safe.
 *
 * @param new_resource If not nullptr, pointer to memory resource to use as new
 * default device memory resource
 * @return device_memory_resource* The previous value of the default device
 * memory resource pointer
 *---------------------------------------------------------------------------**/
device_memory_resource* set_default_resource(
    device_memory_resource* new_resource);

}  // namespace mr
}  // namespace rmm