#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"
#include "fast_alltoall/alltoall_global_scheduler.h"
#include <hip/hip_runtime.h>

// BASELINE
NCCL_API(ncclResult_t, ncclAllToAllv0, uint rankid, uint gpu_n, uint MAX_BUFFER_SIZE_PER_RANK, void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);



ncclResult_t
ncclAllToAllv0_impl(uint rankid, uint gpu_n, uint MAX_BUFFER_SIZE_PER_RANK, void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){

    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));
    ncclResult_t ret, state;

    uint local_rank_id = rankid % gpu_n;
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < rank_n; r++) {
        NCCLCHECK(ncclSend(
            ((char*)sendbuff) + (r * gpu_n + local_rank_id) * MAX_BUFFER_SIZE_PER_RANK *ncclTypeSize(datatype),
            sendcounts[r * gpu_n + local_rank_id],
            datatype,
            r,
            comm,
            stream));
        NCCLCHECK(ncclRecv(
            ((char*)recvbuff) + r * MAX_BUFFER_SIZE_PER_RANK *ncclTypeSize(datatype),
            recvcounts[r],
            datatype,
            r,
            comm,
            stream));
    }

    ret = ncclGroupEnd();
    if (ret == ncclInProgress) {
        do {
        ncclCommGetAsyncError(comm, &state);
        } while (state == ncclInProgress);
    }
    // else if (ret == ncclSuccess) {
    // /* Successfully issued */
    // printf("Rankid: %u, AlltoAll Baseline succeeded\n", rankid);
    // }
    return ncclSuccess;
}


// Proposed algorithm
NCCL_API(ncclResult_t, ncclAllToAllv2, void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);


ncclResult_t
ncclAllToAllv2_impl(void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = sched->rankid,
        local_rank_id = sched->rankid % sched->gpu_n,
        server_id = sched->rankid / sched->gpu_n,
        server_n = sched->server_n,
        gpu_n = sched->gpu_n,
        rankid = sched->rankid,
        MAX_BUFFER_SIZE_PER_RANK = sched->MAX_BUFFER_SIZE_PER_RANK;
    ncclResult_t ret, state;
    // printf("rankid: %u, gpu_n: %u, server_id: %u, server_n: %u\n", rankid, gpu_n, server_id, server_n);

    /* ------------------------------------------------------
        Preparation Stage: Instrinsic AllToAll and Balance
     ----------------------------------------------------- */

    // Instrinsic AllToAll
    NCCLCHECK(ncclGroupStart());
    for (uint r = 0; r < gpu_n; r++){
        uint global_comm_gpu = server_id * gpu_n + r;
        size_t send_data_sz = (sched -> intrinsic_ata)[server_id][local_rank_id * gpu_n + r];
        // printf("rankid: %u, server_id: %u, send to local gpu: %u, data sz: %lu\n", rankid, server_id, r, send_data_sz);
        if (send_data_sz > 0){
            uint sendbuf_offset = global_comm_gpu * gpu_n + local_rank_id;
            void * src_ptr = (char *)sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * ncclTypeSize(datatype);
            NCCLCHECK(ncclSend(
                src_ptr,
                send_data_sz,
                datatype,
                global_comm_gpu,
                comm,
                stream
            ));
            sendpos[sendbuf_offset] += send_data_sz;
        }
        size_t recv_data_sz = (sched -> intrinsic_ata)[server_id][r * gpu_n + local_rank_id];
        if (recv_data_sz > 0){
            void * dst_ptr = (char *)recvbuff + global_comm_gpu * MAX_BUFFER_SIZE_PER_RANK * ncclTypeSize(datatype);
            NCCLCHECK(ncclRecv(
                dst_ptr,
                recv_data_sz,
                datatype,
                global_comm_gpu,
                comm,
                stream
            ));
            recvpos[global_comm_gpu] += recv_data_sz;
        }
    }
    NCCLCHECK(ncclGroupEnd());

    // Load balance
    NCCLCHECK(ncclGroupStart());
    for (uint s = 0; s != server_n; s++){
        if (s == server_id){
            continue;
        }
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
            for (uint channel_id = 0; channel_id < gpu_n; channel_id ++){
                size_t send_data_sz = (sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].sz[channel_id];
                if (send_data_sz > 0){
                    uint global_dst_gpu = s *  gpu_n + channel_id;
                    uint sendbuf_offset = global_dst_gpu * gpu_n + local_rank_id;
                    uint intermediate_gpu_global_id = server_id * gpu_n + local_gpu;
                    void * src_ptr = (char *) sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * ncclTypeSize(datatype) + sendpos[sendbuf_offset] * ncclTypeSize(datatype);
                    NCCLCHECK(ncclSend(
                        src_ptr,
                        send_data_sz,
                        datatype,
                        intermediate_gpu_global_id,
                        comm,
                        stream
                    ));
                    sendpos[sendbuf_offset] += send_data_sz;
                }
                size_t recv_data_sz = (sched -> balance)[server_id][s][local_gpu * gpu_n + local_rank_id].sz[channel_id];
                if (recv_data_sz > 0){
                    uint global_dst_gpu = s *  gpu_n + channel_id;
                    uint sendbuf_offset = global_dst_gpu * gpu_n + local_gpu;
                    uint src_gpu_global_id = server_id * gpu_n + local_gpu;
                    void * dst_ptr = (char *) sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * ncclTypeSize(datatype);
                    NCCLCHECK(ncclRecv(
                        dst_ptr,
                        recv_data_sz,
                        datatype,
                        src_gpu_global_id,
                        comm,
                        stream
                    ));
                    sendcounts [sendbuf_offset] += recv_data_sz;
                }
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());


    /* ------------------------------------------------------
        Pipeline Stage
     ----------------------------------------------------- */

    uint TEMPBUFF_OFFSET = ncclTypeSize(datatype)* gpu_n * gpu_n * MAX_BUFFER_SIZE_PER_RANK;
    uint cur_tempbuff_offset = 0, prev_tempbuff_offset = 0;

    // First step
    struct scheduling_step_t cur_step = (sched -> steps)[0];
    uint dst_server = cur_step.to_server[server_id];
    uint src_server = cur_step.from_server[server_id];
    uint dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
    uint src_gpu_global_id = src_server * gpu_n + local_rank_id;
    uint * channel_send_sched = cur_step.channel[server_id][local_rank_id];
    uint * channel_recv_sched = cur_step.channel[src_server][local_rank_id];

    NCCLCHECK(ncclGroupStart());
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
        for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu++){
            uint send_data_sz = channel_send_sched[local_gpu * gpu_n + from_gpu];
            if (send_data_sz > 0){
                uint global_dst_gpu = dst_server * gpu_n + local_gpu;
                uint sendbuff_offset = global_dst_gpu * gpu_n + from_gpu;
                void * src_ptr = (char *) sendbuff + sendbuff_offset * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + sendpos[sendbuff_offset] * ncclTypeSize(datatype);
                NCCLCHECK(ncclSend(
                    src_ptr,
                    send_data_sz,
                    datatype,
                    dst_gpu_global_id,
                    comm,
                    stream
                ));
                sendpos[sendbuff_offset] += send_data_sz;
            }
            uint recv_data_sz =  channel_recv_sched[local_gpu * gpu_n + from_gpu];
            if (recv_data_sz > 0){
                uint tempbuff_id = local_gpu * gpu_n + from_gpu;
                void * dst_ptr = (char *) tempbuff + cur_tempbuff_offset + tempbuff_id *  ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
                NCCLCHECK(ncclRecv(
                    dst_ptr,
                    recv_data_sz,
                    datatype,
                    src_gpu_global_id,
                    comm,
                    stream
                ));
            }
        }
    }
    ret = ncclGroupEnd();
    if (ret == ncclInProgress) {
        do {
        ncclCommGetAsyncError(comm, &state);
        } while (state == ncclInProgress);
    }
    // else if (ret == ncclSuccess) {
    // /* Successfully issued */
    // printf("Rank %u: step 0 - issue succeeded\n", rankid);
    // }

    // middle steps
    uint prev_dst_server = dst_server,
        prev_src_server = src_server;
    struct scheduling_step_t prev_step = cur_step;
    struct recv_data_t * restore_send_sched, * restore_recv_sched, * dcopy_sched;
    uint step_n = sched -> step_n;

    for (uint step_id = 1; step_id < step_n - 1; step_id ++){
        cur_step = (sched -> steps)[step_id];
        dst_server = cur_step.to_server[server_id];
        src_server = cur_step.from_server[server_id];

        dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
        src_gpu_global_id = src_server * gpu_n + local_rank_id;

        channel_send_sched = cur_step.channel[server_id][local_rank_id];
        channel_recv_sched = cur_step.channel[src_server][local_rank_id];

        restore_send_sched = cur_step.restore[prev_src_server][local_rank_id];
        dcopy_sched = cur_step.direct_cpy[prev_src_server][local_rank_id];

        cur_tempbuff_offset = (step_id % 2 == 1) ? TEMPBUFF_OFFSET : 0;
        prev_tempbuff_offset = ((step_n - 1) % 2 == 1) ? 0 : TEMPBUFF_OFFSET;

        NCCLCHECK(ncclGroupStart());
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
            for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu++){
                uint send_data_sz = channel_send_sched[local_gpu * gpu_n + from_gpu];
                if (send_data_sz > 0){
                    uint global_dst_gpu = dst_server * gpu_n + local_gpu;
                    uint sendbuff_offset = global_dst_gpu * gpu_n + from_gpu;
                    void * src_ptr = (char *) sendbuff + sendbuff_offset * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + sendpos[sendbuff_offset] * ncclTypeSize(datatype);
                    NCCLCHECK(ncclSend(
                        src_ptr,
                        send_data_sz,
                        datatype,
                        dst_gpu_global_id,
                        comm,
                        stream
                    ));
                    sendpos[sendbuff_offset] += send_data_sz;
                }
                uint recv_data_sz =  channel_recv_sched[local_gpu * gpu_n + from_gpu];
                if (recv_data_sz > 0){
                    uint tempbuff_id = local_gpu * gpu_n + from_gpu;
                    void * dst_ptr = (char *) tempbuff + cur_tempbuff_offset + tempbuff_id *  ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
                    NCCLCHECK(ncclRecv(
                        dst_ptr,
                        recv_data_sz,
                        datatype,
                        src_gpu_global_id,
                        comm,
                        stream
                    ));
                }
            }
        }


        // restore
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
            if (local_gpu == local_rank_id){
                continue;
            }
            for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
                uint send_data_sz = restore_send_sched[local_gpu * gpu_n + from_gpu].sz;
                if (send_data_sz > 0){
                    dst_gpu_global_id = server_id * gpu_n + local_gpu;
                    uint src_gpu_tempbuff_id = local_gpu * gpu_n + from_gpu;
                    void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
                    NCCLCHECK(ncclSend(
                        src_ptr,
                        send_data_sz,
                        datatype,
                        dst_gpu_global_id,
                        comm,
                        stream
                    ));
                }
                restore_recv_sched = cur_step.restore[prev_src_server][local_gpu];
                uint recv_data_sz = restore_recv_sched[local_rank_id * gpu_n + from_gpu].sz;
                if (recv_data_sz > 0){
                    src_gpu_global_id = prev_src_server * gpu_n + from_gpu;
                    uint intermediate_gpu_global_id = server_id * gpu_n + local_gpu;
                    void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + restore_recv_sched[local_rank_id * gpu_n + from_gpu].offset * ncclTypeSize(datatype);
                    NCCLCHECK(ncclRecv(
                        dst_ptr,
                        recv_data_sz,
                        datatype,
                        intermediate_gpu_global_id,
                        comm,
                        stream
                    ));
                    recvpos[src_gpu_global_id] += recv_data_sz;
                }
            }
        }


        //direct cpy
        for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
            if (dcopy_sched[from_gpu].sz > 0){
                src_gpu_global_id = prev_src_server  * gpu_n + from_gpu;
                uint src_gpu_tempbuff_id = local_rank_id * gpu_n + from_gpu;
                void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
                void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * ncclTypeSize(datatype);
                NCCLCHECK(ncclSend(
                    src_ptr,
                    dcopy_sched[from_gpu].sz,
                    datatype,
                    rankid,
                    comm,
                    stream
                ));
                NCCLCHECK(ncclRecv(
                    dst_ptr,
                    dcopy_sched[from_gpu].sz,
                    datatype,
                    rankid,
                    comm,
                    stream
                ));
                recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
            }
        }

        ret = ncclGroupEnd();
        if (ret == ncclInProgress) {
            do {
            ncclCommGetAsyncError(comm, &state);
            } while (state == ncclInProgress);
        }
        // else if (ret == ncclSuccess) {
        // /* Successfully issued */
        // printf("Rank %u: step %u - issue succeeded\n", rankid, step_id);
        // }

        prev_src_server = src_server;
        prev_dst_server = dst_server;
    }

    // last restore
    prev_tempbuff_offset = ((step_n - 1) % 2 == 1) ? 0 : TEMPBUFF_OFFSET;
    cur_step = (sched -> steps)[step_n - 1];
    restore_send_sched = cur_step.restore[prev_src_server][local_rank_id];
    dcopy_sched = cur_step.direct_cpy[prev_src_server][local_rank_id];

    // // direct cpy
    // for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
    //     if (dcopy_sched[from_gpu].sz > 0){
    //         src_gpu_global_id = prev_src_server  * gpu_n + from_gpu;
    //         uint src_gpu_tempbuff_id = local_rank_id * gpu_n + from_gpu;
    //         void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
    //         void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * ncclTypeSize(datatype);
    //         hipMemcpy(dst_ptr, src_ptr, ncclTypeSize(datatype) * dcopy_sched[from_gpu].sz, hipMemcpyDeviceToDevice);
    //         recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
    //     }
    // }

    // restore
    NCCLCHECK(ncclGroupStart());
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
        if (local_gpu == local_rank_id){
            continue;
        }
        for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
            uint send_data_sz = restore_send_sched[local_gpu * gpu_n + from_gpu].sz;
            if (send_data_sz > 0){
                dst_gpu_global_id = server_id * gpu_n + local_gpu;
                uint src_gpu_tempbuff_id = local_gpu * gpu_n + from_gpu;
                void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
                NCCLCHECK(ncclSend(
                    src_ptr,
                    send_data_sz,
                    datatype,
                    dst_gpu_global_id,
                    comm,
                    stream
                ));
            }
            restore_recv_sched = cur_step.restore[prev_src_server][local_gpu];
            uint recv_data_sz = restore_recv_sched[local_rank_id * gpu_n + from_gpu].sz;
            if (recv_data_sz > 0){
                src_gpu_global_id = prev_src_server * gpu_n + from_gpu;
                uint intermediate_gpu_global_id = server_id * gpu_n + local_gpu;
                void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + restore_recv_sched[local_rank_id * gpu_n + from_gpu].offset * ncclTypeSize(datatype);
                NCCLCHECK(ncclRecv(
                    dst_ptr,
                    recv_data_sz,
                    datatype,
                    intermediate_gpu_global_id,
                    comm,
                    stream
                ));
                recvpos[src_gpu_global_id] += recv_data_sz;
            }
        }
    }

    // direct cpy
    for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
        if (dcopy_sched[from_gpu].sz > 0){
            src_gpu_global_id = prev_src_server * gpu_n + from_gpu;
            uint src_gpu_tempbuff_id = local_rank_id * gpu_n + from_gpu;
            void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
            void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * ncclTypeSize(datatype);
            NCCLCHECK(ncclSend(
                    src_ptr,
                    dcopy_sched[from_gpu].sz,
                    datatype,
                    rankid,
                    comm,
                    stream
                ));
                NCCLCHECK(ncclRecv(
                    dst_ptr,
                    dcopy_sched[from_gpu].sz,
                    datatype,
                    rankid,
                    comm,
                    stream
                ));
            recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
        }
    }
    NCCLCHECK(ncclGroupEnd());

    return ncclSuccess;
}