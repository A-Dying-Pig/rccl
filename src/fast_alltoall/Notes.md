## Register new api in rccl
In `src/include/api_trace.h`, do the following steps:
* Define new function type by adding:
```cpp
typedef ncclResult_t <api_name>_fn_t (<api_parameters>);

// for example
typedef ncclResult_t (*ncclAllToAllv2_fn_t)(
    const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void* recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
```

* Register the defined api by adding the following code to `rcclApiFuncTable`:
```cpp
<api_name>_fn_t                <api_name>_fn;
//for example
ncclAllToAllv2_fn_t            ncclAllToAllv2_fn;
```

In `src/misc/api_trace.cc`, do the following steps:
* Define an api implementation by adding:
```cpp
ncclResult_t <api_name>_impl(<api_parameters>);
//for example
ncclResult_t
ncclAllToAllv_impl(const void* sendbuff, const size_t sendcounts[],
                   const size_t sdispls[], void* recvbuff, const size_t recvcounts[],
                   const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm,
                   hipStream_t stream);
```

* Assign an index to the new api in the api table:
```cpp
RCCL_ASSERT_OFFSET(rcclApiFuncTable, <api_name>_fn, <idx>);
// for example
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllToAllv_fn, 37);
```
Add implementation function's pointer to `rcclApiFuncTable`; the pointer's location is its index:
```cpp
rcclApiFuncTable{sizeof(rcclApiFuncTable),
                &ncclAllGather_impl,
                &ncclAllReduce_impl,
                ...,
                &<api_name>_impl};
```
* Declare new api
```cpp 
NCCL_API(ncclResult_t, <api_name>, <api_parameters>);
//for example
NCCL_API(ncclResult_t, ncclAllToAllv2, const void* sendbuff, const size_t sendcounts[],
         const size_t sdispls[], void* recvbuff, const size_t recvcounts[],
         const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm,
         hipStream_t stream);
```
and 
```cpp
ncclResult_t <api_name>(<api_parameters>){
    return ::rccl::RcclGetFunctionTable()-><api_name>_fn (<api_parameters>);
}
//for example
ncclResult_t
ncclAllToAllv2(const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
              void* recvbuff, const size_t recvcounts[], const size_t rdispls[],
              ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclAllToAllv2_fn(sendbuff, sendcounts, sdispls,
                                                            recvbuff, recvcounts, rdispls,
                                                            datatype, comm, stream);
}
```
* In any `.c` implementation file, include `api_trace.h`, declare NCCL_API, and implement the api:
```cpp
NCCL_API(ncclResult_t, <api_name>, <api_parameters>);
ncclResult_t <api_name>_impl(<api_parameters>){
// implementation
// ...
};
//for example
NCCL_API(ncclResult_t, ncclAllToAllv2, const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
ncclResult_t
ncclAllToAllv2_impl(const void* sendbuff, const size_t sendcounts[],
                   const size_t sdispls[], void* recvbuff, const size_t recvcounts[],
                   const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm,
                   hipStream_t stream){
// ...                    
}
```

The api function is executed separately in parallel at each GPU.