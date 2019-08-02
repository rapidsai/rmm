===============================================
RMMEP-1: Implement deferrable memory management
===============================================

:Author: Peter Andreas Entschev <pentschev@nvidia.com>

Abstract
========

We propose the implementation of a deferrable memory management scheme,
allowing memory buffers to be moved to different storage media (e.g., host
memory) while preventing the code that initially requested the memory buffer
from accessing the invalid memory that a pointer would now point to.

Detailed description
====================

Large problems require large amounts of memory, that are often not available in
a single device. This type of problem usually is solved by moving data to a
device with larger memory capacity (and often slower). Some libraries such as
dask-distributed and dask-cuda solve this problem directly on the Python side as
memory objects can be requested online without the risk of referring to an
invalid memory pointer, but this can be considerably more complicated on C/C++
or CUDA, due to the explicit usage of pointers addressing memory directly.

What we propose here is a new structure for RMM that would hold a minimal amount
of information about memory buffers that are requested by application code (such
as cuDF), including whether that buffer is available for immediate use or has
been deferred to another memory device and would require that buffer to be
retrieved first.

The structure itself is not sufficient, therefore, some extra functionality is
required to synchronize memory beforehand. There are two main requirements for
such functions:

1) Retrieve memory before addressing it;

2) Locking memory when it's being accessed (e.g., during a CUDA kernel call) to
prevent it from being deferred during execution, and releasing the lock when
execution has finished.

This information must be later passed to a central Python controller also
proposed here, which will allow applications on the Python side, such as Dask,
to move the data safely take control of each memory object independently,
giving it the capability to defer them to different storage media.

The Python controller will have a callback mechanism that allows C++ code to
register when new memory object is allocated or freed. After this, both C++ and
Python code will be capable of acessing these memory regions, and with the
maintenance of additional signals, both sides are capable of keeping safety,
locking memory on C++ so that Python won't defer it during usage, as well as
informing C++ that a memory buffer was deferred and it has to wait for Python
to retrieve it back before it can be used.

Implementation
==============

Memory structure
----------------

The memory structure is intended to be very simple and hold only a small amount
of information. It would look something like:

.. code-block:: c++

    typedef size_t hash_t;

    typedef struct
    {
        void * ptr;
        size_t size;
        hash_t hashID;
        cudaStream_t stream;
        atomic<bool> isDeferred;
        atomic<bool> isLocked;
    } DeferrableMemory;

Locking mechanism
-----------------

We may start by defining generic locking and unlocking mechanisms, both taking
an arbitrary number of arguments, the reason for this will become clear later.
The two mechanisms may be used for both execution cases to follow: synchronous
and asynchronous.

.. code-block:: c++

    void retrieveAndLockDeferrables()
    {
        return;
    }

    template<typename T, typename... Targs>
    void retrieveAndLockDeferrables(T t, Targs... Fargs)
    {
        if (std::is_same<T, std::shared_ptr<DeferrableMemory>>::value)
        {
            retrieveFromPython(t);
            t.isLocked = true;
        }
        retrieveAndLockDeferrables(Fargs...);
    }

    void unlockDeferrables()
    {
        return;
    }

    template<typename T, typename... Targs>
    void unlockDeferrables(T t, Targs... Fargs)
    {
        if (std::is_same<T, std::shared_ptr<DeferrableMemory>>::value)
        {
            t.isLocked = false;
        }
        unlockDeferrables(Fargs...);
    }

Synchronous execution (default stream)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synchronous case is very simple to solve. A possible locking mechanism usage
is to make a wrapper function which would receive any number of objects,
then lock all the DeferrableMemory objects, launch the synchronous CUDA function
and unlock it at the end. A simple wrapper function like the one below would
suffice.

.. code-block:: c++

    template<typename Lambda, typename... Targs>
    void lockAndLaunch(Lambda&& func, Targs... Fargs)
    {
        retrieveAndLockDeferrable(Fargs...);
        std::forward<Lambda>(func)(Fargs...)
        unlockDeferrable(Fargs...);
    }

Note here that the function func is not a CUDA API or kernel call, but a lambda.
One may thus write a full function that executes various operations, without a
need for this deferrable memory mechanism to know any of its implementation
details.

Asynchronous execution (non-default stream)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The asynchronous execution case is considerably more complicated than the
synchronous one. If we used lockAndLaunch, operations on memory may still be
ongoing, but the DeferrableMemory objects would immediately be unlocked, so
there would be no way to know when such an object is being used somewhere or
idle. That said, DeferrableMemory objects cannot be unlocked at the end of the
launch function, but only when a synchronization operation is executed.

To solve such a situation, we must keep track of what DeferrableMemory objects
are in use in a given stream. We can do that bookkeeping with a singleton class
that operates a map of streams to lists of such objects.

.. code-block:: c++

    class DeferrableMemoryMap
    {
        public:
            static DeferrableMemoryMap& getInstance()
            {
                static DeferrableMemoryMap instance;
                return instance;
            }

            void registerDeferrableMemory(cudaStream_t stream,
                                          std::shared_ptr<DeferrableMemory>& deferrable)
            {
                _streamDeferrableMap[stream].push_back(deferrable);
            }

            void clearStreamMap(cudaStream_t stream)
            {
                _streamDeferrableMap[stream].clear();
            }

            DeferrableMemoryMap(DeferrableMemoryMap const&) = delete;
            DeferrableMemoryMap& operator=(DeferrableMemoryMap const&) = delete;

        private:
            DeferrableMemoryMap() {}

            std::map<cudaStream_t, std::vector<std::shared_ptr<DeferrableMemory>>>
                _streamDeferrableMap;
    }

We can now define functions for locking and unlocking mechanisms, given the
streams. The first function will simply set the isLocked attribute and push a
reference to that memory object into the map defined above. The unlocking
function will traverse the list of memory objects locked into that stream,
unsetting the isLocked attribute and clearing the list at the end.

.. code-block:: c++

    void retrieveAndLockDeferrablesStream(cudaStream_t stream)
    {
        return;
    }

    template<typename T, typename... Targs>
    void retrieveAndLockDeferrablesStream(cudaStream_t stream, T t, Targs... Fargs)
    {
        if (std::is_same<T, std::shared_ptr<DeferrableMemory>>::value)
        {
            retrieveFromPython(t);
            t.isLocked = true;
            deferrableMap.registerDeferrableMemory(stream, deferrableMemory);
        }
        retrieveAndLockDeferrablesStream(cudaStream_t stream, Fargs...);
    }

    void unlockDeferrableStream(cudaStream_t stream)
    {
        for (auto& deferrableMemory: streamDeferrableMap[stream])
        {
            deferrableMemory.isLocked = false;
        }
        auto& deferrableMap = DeferrableMemoryMap::getInstance();
        streamDeferrableMap[stream].clear();
    }

We now have all the tools needed to launch asynchronous CUDA calls. The
prototype is very similar to the synchronous lockAndLaunch, but now we need
also a stream and there will not be any unlocking done at the end, which will be
done opportunistically on a different call.

.. code-block:: c++

    template<typename Lambda, typename... Targs>
    void lockAndLaunchAsync(Lambda&& func, Targs... Fargs, cudaStream_t stream)
    {
        retrieveAndLockDeferrablesStream(stream, Fargs...);
        std::forward<Lambda>(func)(stream, Fargs...)
    }

    void syncAndUnlock(cudaStream_t stream)
    {
        cudaStreamSynchronize(stream);
        unlockDeferrableStream(stream);
    }

Python Interface
----------------

TODO, needs to cover:

- DeferrableMemory registration
- DeferrableMemory deallocation/deferred signaling
- DeferrableMemory retrieval
- Interface for Python client (e.g. Dask)
