---
slug: api-reference/cpp-api-errors
---

# Errors

Generated from RMM C++ headers.

## `cpp/include/rmm/error.hpp`

### Logic Error Struct

Exception thrown when logical precondition is violated.

```cpp
struct logic_error : public std::logic_error {
```

_Source: `cpp/include/rmm/error.hpp:24`_

### CUDA Error Struct

Exception thrown when a CUDA error is encountered.

```cpp
struct cuda_error : public std::runtime_error {
```

_Source: `cpp/include/rmm/error.hpp:34`_

### Bad Alloc Class

Exception thrown when an RMM allocation fails

```cpp
class bad_alloc : public std::bad_alloc {
```

_Source: `cpp/include/rmm/error.hpp:44`_

### Bad Alloc Constructor (error.hpp:51)

Constructs a bad_alloc with the error message.

```cpp
bad_alloc(const char* msg);
```

_Source: `cpp/include/rmm/error.hpp:51`_

### Bad Alloc Constructor (error.hpp:58)

Constructs a bad_alloc with the error message.

```cpp
bad_alloc(std::string const& msg);
```

_Source: `cpp/include/rmm/error.hpp:58`_

### Out Of Memory Class

Exception thrown when RMM runs out of memory

```cpp
class out_of_memory : public bad_alloc {
```

_Source: `cpp/include/rmm/error.hpp:76`_

### Out Of Memory Constructor (error.hpp:83)

Constructs an out_of_memory with the error message.

```cpp
out_of_memory(const char* msg);
```

_Source: `cpp/include/rmm/error.hpp:83`_

### Out Of Memory Constructor (error.hpp:90)

Constructs an out_of_memory with the error message.

```cpp
out_of_memory(std::string const& msg);
```

_Source: `cpp/include/rmm/error.hpp:90`_

### Out Of Range Class

Exception thrown when attempting to access outside of a defined range

```cpp
class out_of_range : public std::out_of_range {
```

_Source: `cpp/include/rmm/error.hpp:99`_

### Invalid Argument Class

Exception thrown when an argument to a function is invalid

```cpp
class invalid_argument : public std::invalid_argument {
```

_Source: `cpp/include/rmm/error.hpp:108`_
