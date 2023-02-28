# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
try:
    from torch.cuda.memory import CUDAPluggableAllocator
except ImportError:
    rmm_torch_allocator = None
else:
    import rmm._lib.torch_allocator

    _alloc_free_lib_path = rmm._lib.torch_allocator.__file__
    rmm_torch_allocator = CUDAPluggableAllocator(
        _alloc_free_lib_path,
        alloc_fn_name="allocate",
        free_fn_name="deallocate",
    )
