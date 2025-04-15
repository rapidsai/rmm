# Copyright (c) 2022, NVIDIA CORPORATION.
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
#

import gdb


class HostIterator:
    """Iterates over arrays in host memory."""

    def __init__(self, start, size):
        self.item = start
        self.size = size
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        elt = self.item.dereference()
        count = self.count
        self.item += 1
        self.count += 1
        return (f"[{count}]", elt)


class DeviceIterator:
    """Iterates over device arrays by copying chunks to the host."""

    def __init__(self, start, size):
        self.exec = exec
        self.item = start
        self.size = size
        self.count = 0
        self.buffer = None
        self.sizeof = self.item.dereference().type.sizeof
        self.buffer_start = 0
        # At most 1 MB or size, at least 1
        self.buffer_size = min(size, max(1, 2**20 // self.sizeof))
        self.buffer = gdb.parse_and_eval(
            f"(void*)malloc({self.buffer_size * self.sizeof})"
        )
        self.buffer.fetch_lazy()
        self.buffer_count = self.buffer_size
        self.update_buffer()

    def update_buffer(self):
        if self.buffer_count >= self.buffer_size:
            self.buffer_item = gdb.parse_and_eval(hex(self.buffer)).cast(
                self.item.type
            )
            self.buffer_count = 0
            self.buffer_start = self.count
            device_addr = hex(self.item.dereference().address)
            buffer_addr = hex(self.buffer)
            size = (
                min(self.buffer_size, self.size - self.buffer_start)
                * self.sizeof
            )
            status = gdb.parse_and_eval(
                f"(cudaError)cudaMemcpy({buffer_addr}, {device_addr}, {size}, "
                "cudaMemcpyDeviceToHost)"
            )
            if status != 0:
                raise gdb.MemoryError(f"memcpy from device failed: {status}")

    def __del__(self):
        gdb.parse_and_eval(f"(void)free({hex(self.buffer)})").fetch_lazy()

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        self.update_buffer()
        elt = self.buffer_item.dereference()
        self.buffer_item += 1
        self.buffer_count += 1
        count = self.count
        self.item += 1
        self.count += 1
        return (f"[{count}]", elt)


class RmmDeviceUVectorPrinter(gdb.printing.PrettyPrinter):
    """Print a rmm::device_uvector."""

    def __init__(self, val):
        self.val = val
        el_type = val.type.template_argument(0)
        self.pointer = val["_storage"]["_data"].cast(el_type.pointer())
        self.size = int(val["_storage"]["_size"]) // el_type.sizeof
        self.capacity = int(val["_storage"]["_capacity"]) // el_type.sizeof

    def children(self):
        return DeviceIterator(self.pointer, self.size)

    def to_string(self):
        return (
            f"{self.val.type} of length {self.size}, capacity {self.capacity}"
        )

    def display_hint(self):
        return "array"


# Workaround to avoid using the pretty printer on things like
# std::vector<int>::iterator
def is_template_type_not_alias(typename):
    loc = typename.find("<")
    if loc is None:
        return False
    depth = 0
    for char in typename[loc:-1]:
        if char == "<":
            depth += 1
        if char == ">":
            depth -= 1
        if depth == 0:
            return False
    return True


def template_match(typename, template_name):
    return typename.startswith(template_name + "<") and typename.endswith(">")


def lookup_rmm_type(val):
    if not str(val.type.unqualified()).startswith("rmm::"):
        return None
    suffix = str(val.type.unqualified())[5:]
    if not is_template_type_not_alias(suffix):
        return None
    if template_match(suffix, "device_uvector"):
        return RmmDeviceUVectorPrinter(val)
    return None


gdb.pretty_printers.append(lookup_rmm_type)
