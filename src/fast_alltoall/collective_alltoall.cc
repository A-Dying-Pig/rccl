#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"
#include "fast_alltoall/alltoall_global_scheduler.h"
#include <hip/hip_runtime.h>



NCCL_API(ncclResult_t, ncclAllToAllv2, uint rankid, void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, void* syncbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);



ncclResult_t
ncclAllToAllv2_impl(uint rankid, void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, void* syncbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = rankid,
        local_rank_id = rankid % rank_n,
        server_id = rankid / GPU_NUM_PER_SERVER,
        server_n = (rank_n + GPU_NUM_PER_SERVER - 1) /  GPU_NUM_PER_SERVER;
    ncclResult_t ret, state;
    printf("rankid: %u, server_id: %u, server_n: %u\n", rankid, server_id, server_n);

    /* ------------------------------------------------------
        Preparation Stage: Instrinsic AllToAll and Balance
     ----------------------------------------------------- */

    // Instrinsic AllToAll
    NCCLCHECK(ncclGroupStart());
    for (uint r = 0; r < GPU_NUM_PER_SERVER; r++){
        uint global_comm_gpu = server_id * GPU_NUM_PER_SERVER + r;
        size_t send_data_sz = (sched -> intrinsic_ata)[server_id][local_rank_id * GPU_NUM_PER_SERVER + r];
        printf("rankid: %u, server_id: %u, send to local gpu: %u, data sz: %lu\n", rankid, server_id, r, send_data_sz);
        if (send_data_sz > 0){
            uint sendbuf_offset = global_comm_gpu * GPU_NUM_PER_SERVER + local_rank_id;
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
        size_t recv_data_sz = (sched -> intrinsic_ata)[server_id][r * GPU_NUM_PER_SERVER + local_rank_id];
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
    ret = ncclGroupEnd();
    if (ret == ncclInProgress) {
        do {
        ncclCommGetAsyncError(comm, &state);
        } while (state == ncclInProgress);
    } else if (ret == ncclSuccess) {
    /* Successfully issued */
    printf("Rankid: %u, Kernel - Intrinsic AlltoAll - issue succeeded\n", rankid);
    }

    // Load balance
    for (uint s = 0; s != server_n; s++){
        if (s == server_id){
            continue;
        }
        NCCLCHECK(ncclGroupStart());
        for (uint local_gpu = 0; local_gpu < GPU_NUM_PER_SERVER; local_gpu++){
            for (uint channel_id = 0; channel_id < GPU_NUM_PER_SERVER; channel_id ++){
                size_t send_data_sz = (sched -> balance)[server_id][s][local_rank_id * GPU_NUM_PER_SERVER + local_gpu].sz[channel_id];
                if (send_data_sz > 0){
                    uint global_dst_gpu = s *  GPU_NUM_PER_SERVER + channel_id;
                    uint sendbuf_offset = global_dst_gpu * GPU_NUM_PER_SERVER + local_rank_id;
                    uint intermediate_gpu_global_id = server_id * GPU_NUM_PER_SERVER + local_gpu;
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
                size_t recv_data_sz = (sched -> balance)[server_id][s][local_gpu * GPU_NUM_PER_SERVER + local_rank_id].sz[channel_id];
                if (recv_data_sz > 0){
                    uint global_dst_gpu = s *  GPU_NUM_PER_SERVER + channel_id;
                    uint sendbuf_offset = global_dst_gpu * GPU_NUM_PER_SERVER + local_gpu;
                    uint src_gpu_global_id = server_id * GPU_NUM_PER_SERVER + local_gpu;
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
        ret = ncclGroupEnd();
        if (ret == ncclInProgress) {
            do {
            ncclCommGetAsyncError(comm, &state);
            } while (state == ncclInProgress);
        } else if (ret == ncclSuccess) {
        /* Successfully issued */
        printf("Kernel - Load Balance from server %u to server %u - issue succeeded\n", server_id, s);
        }
    }

    /* ------------------------------------------------------
        Pipeline Stage
     ----------------------------------------------------- */

    uint TEMPBUFF_OFFSET = ncclTypeSize(datatype)* GPU_NUM_PER_SERVER * GPU_NUM_PER_SERVER * MAX_BUFFER_SIZE_PER_RANK;
    uint cur_tempbuff_offset = 0, prev_tempbuff_offset = 0;

    // First step
    struct scheduling_step_t cur_step = (sched -> steps)[0];
    uint dst_server = cur_step.to_server[server_id];
    uint src_server = cur_step.from_server[server_id];
    uint dst_gpu_global_id = dst_server * GPU_NUM_PER_SERVER + local_rank_id;
    uint src_gpu_global_id = src_server * GPU_NUM_PER_SERVER + local_rank_id;
    uint * channel_send_sched = cur_step.channel[server_id][local_rank_id];
    uint * channel_recv_sched = cur_step.channel[src_server][local_rank_id];

    NCCLCHECK(ncclGroupStart());
    for (uint local_gpu = 0; local_gpu < GPU_NUM_PER_SERVER; local_gpu++){
        for (uint from_gpu = 0; from_gpu < GPU_NUM_PER_SERVER; from_gpu++){
            uint send_data_sz = channel_send_sched[local_gpu * GPU_NUM_PER_SERVER + from_gpu];
            if (send_data_sz > 0){
                uint global_dst_gpu = dst_server * GPU_NUM_PER_SERVER + local_gpu;
                uint sendbuff_offset = global_dst_gpu * GPU_NUM_PER_SERVER + from_gpu;
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
            uint recv_data_sz =  channel_recv_sched[local_gpu * GPU_NUM_PER_SERVER + from_gpu];
            if (recv_data_sz > 0){
                uint tempbuff_id = local_gpu * GPU_NUM_PER_SERVER + from_gpu;
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
    } else if (ret == ncclSuccess) {
    /* Successfully issued */
    printf("Rank %u: Kernel - step 0 - issue succeeded\n", rankid);
    }

    // // middle steps
    uint prev_dst_server = dst_server,
        prev_src_server = src_server;
    struct scheduling_step_t prev_step = cur_step;
    struct recv_data_t * restore_send_sched, * restore_recv_sched, * dcopy_sched;
    uint step_n = sched -> step_n;

    // for (uint step_id = 1; step_id < step_n - 1; step_id ++){
    //     cur_step = (sched -> steps)[step_id];
    //     dst_server = cur_step.to_server[server_id];
    //     src_server = cur_step.from_server[server_id];

    //     dst_gpu_global_id = dst_server * GPU_NUM_PER_SERVER + local_rank_id;
    //     src_gpu_global_id = src_server * GPU_NUM_PER_SERVER + local_rank_id;

    //     channel_send_sched = cur_step.channel[server_id][local_rank_id];
    //     channel_recv_sched = cur_step.channel[src_server][local_rank_id];

    //     restore_send_sched = cur_step.restore[prev_src_server][local_rank_id];
    //     dcopy_sched = cur_step.direct_cpy[prev_src_server][local_rank_id];

    //     cur_tempbuff_offset = (step_id % 2 == 1) ? TEMPBUFF_OFFSET : 0;
    //     prev_tempbuff_offset = ((step_n - 1) % 2 == 1) ? 0 : TEMPBUFF_OFFSET;

    //     NCCLCHECK(ncclGroupStart());
    //     for (uint local_gpu = 0; local_gpu < GPU_NUM_PER_SERVER; local_gpu++){
    //         for (uint from_gpu = 0; from_gpu < GPU_NUM_PER_SERVER; from_gpu++){
    //             uint send_data_sz = channel_send_sched[local_gpu * GPU_NUM_PER_SERVER + from_gpu];
    //             if (send_data_sz > 0){
    //                 uint global_dst_gpu = dst_server * GPU_NUM_PER_SERVER + local_gpu;
    //                 uint sendbuff_offset = global_dst_gpu * GPU_NUM_PER_SERVER + from_gpu;
    //                 void * src_ptr = (char *) sendbuff + sendbuff_offset * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + sendpos[sendbuff_offset] * ncclTypeSize(datatype);
    //                 NCCLCHECK(ncclSend(
    //                     src_ptr,
    //                     send_data_sz,
    //                     datatype,
    //                     dst_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //                 sendpos[sendbuff_offset] += send_data_sz;
    //             }
    //             uint recv_data_sz =  channel_recv_sched[local_gpu * GPU_NUM_PER_SERVER + from_gpu];
    //             if (recv_data_sz > 0){
    //                 uint tempbuff_id = local_gpu * GPU_NUM_PER_SERVER + from_gpu;
    //                 void * dst_ptr = (char *) tempbuff + cur_tempbuff_offset + tempbuff_id *  ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
    //                 NCCLCHECK(ncclRecv(
    //                     dst_ptr,
    //                     recv_data_sz,
    //                     datatype,
    //                     src_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //             }
    //         }
    //     }


    //     // restore
    //     for (uint local_gpu = 0; local_gpu < GPU_NUM_PER_SERVER; local_gpu++){
    //         if (local_gpu == local_rank_id){
    //             continue;
    //         }
    //         for (uint from_gpu = 0; from_gpu < GPU_NUM_PER_SERVER; from_gpu ++){
    //             uint send_data_sz = restore_send_sched[local_gpu * GPU_NUM_PER_SERVER + from_gpu].sz;
    //             if (send_data_sz > 0){
    //                 dst_gpu_global_id = server_id * GPU_NUM_PER_SERVER + local_gpu;
    //                 uint src_gpu_tempbuff_id = local_rank_id * GPU_NUM_PER_SERVER + from_gpu;
    //                 void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
    //                 NCCLCHECK(ncclSend(
    //                     src_ptr,
    //                     send_data_sz,
    //                     datatype,
    //                     dst_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //             }
    //             restore_recv_sched = cur_step.restore[prev_src_server][local_gpu];
    //             uint recv_data_sz = restore_recv_sched[local_rank_id * GPU_NUM_PER_SERVER + from_gpu].sz;
    //             if (recv_data_sz > 0){
    //                 src_gpu_global_id = prev_src_server * GPU_NUM_PER_SERVER + from_gpu;
    //                 uint intermediate_gpu_global_id = server_id * GPU_NUM_PER_SERVER + local_gpu;
    //                 void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + restore_recv_sched[local_rank_id * GPU_NUM_PER_SERVER + from_gpu].offset * ncclTypeSize(datatype);
    //                 NCCLCHECK(ncclRecv(
    //                     dst_ptr,
    //                     recv_data_sz,
    //                     datatype,
    //                     intermediate_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //                 recvpos[src_gpu_global_id] += recv_data_sz;
    //             }
    //         }
    //     }
    //     NCCLCHECK(ncclGroupEnd());


    //     // direct cpy
    //     for (uint from_gpu = 0; from_gpu < GPU_NUM_PER_SERVER; from_gpu ++){
    //         if (dcopy_sched[from_gpu].sz > 0){
    //             src_gpu_global_id = prev_src_server  * GPU_NUM_PER_SERVER + from_gpu;
    //             uint src_gpu_tempbuff_id = local_rank_id * GPU_NUM_PER_SERVER + from_gpu;
    //             void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
    //             void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * ncclTypeSize(datatype);
    //             hipMemcpy(dst_ptr, src_ptr, ncclTypeSize(datatype) * dcopy_sched[from_gpu].sz, hipMemcpyDeviceToDevice);
    //             recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
    //         }
    //     }


    //     prev_src_server = src_server;
    //     prev_dst_server = dst_server;

    //     // Sync all the GPUs to move to the next step
    //     NCCLCHECK(ncclGroupStart());
    //     for (int r = 0; r < rank_n; r++) {
    //         NCCLCHECK(ncclSend(
    //             ((char*)syncbuff) + r * ncclTypeSize(datatype),
    //             1,
    //             datatype,
    //             r,
    //             comm,
    //             stream));
    //         NCCLCHECK(ncclRecv(
    //             ((char*)syncbuff) + r * 2 * ncclTypeSize(datatype),
    //             1,
    //             datatype,
    //             r,
    //             comm,
    //             stream));
    //     }
    //     NCCLCHECK(ncclGroupEnd());

    // }

    // last restore
    prev_tempbuff_offset = ((step_n - 1) % 2 == 1) ? 0 : TEMPBUFF_OFFSET;
    cur_step = (sched -> steps)[step_n - 1];
    restore_send_sched = cur_step.restore[prev_src_server][local_rank_id];
    dcopy_sched = cur_step.direct_cpy[prev_src_server][local_rank_id];

    // direct cpy
    for (uint from_gpu = 0; from_gpu < GPU_NUM_PER_SERVER; from_gpu ++){
        if (dcopy_sched[from_gpu].sz > 0){
            src_gpu_global_id = prev_src_server  * GPU_NUM_PER_SERVER + from_gpu;
            uint src_gpu_tempbuff_id = local_rank_id * GPU_NUM_PER_SERVER + from_gpu;
            void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK;
            void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * ncclTypeSize(datatype);
            hipMemcpy(dst_ptr, src_ptr, ncclTypeSize(datatype) * dcopy_sched[from_gpu].sz, hipMemcpyDeviceToDevice);
            recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
        }
    }

    // restore
    NCCLCHECK(ncclGroupStart());
    for (uint local_gpu = 0; local_gpu < GPU_NUM_PER_SERVER; local_gpu++){
        if (local_gpu == local_rank_id){
            continue;
        }
        for (uint from_gpu = 0; from_gpu < GPU_NUM_PER_SERVER; from_gpu ++){
            uint send_data_sz = restore_send_sched[local_gpu * GPU_NUM_PER_SERVER + from_gpu].sz;
            if (send_data_sz > 0){
                dst_gpu_global_id = server_id * GPU_NUM_PER_SERVER + local_gpu;
                uint src_gpu_tempbuff_id = local_rank_id * GPU_NUM_PER_SERVER + from_gpu;
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
            uint recv_data_sz = restore_recv_sched[local_rank_id * GPU_NUM_PER_SERVER + from_gpu].sz;
            if (recv_data_sz > 0){
                src_gpu_global_id = prev_src_server * GPU_NUM_PER_SERVER + from_gpu;
                uint intermediate_gpu_global_id = server_id * GPU_NUM_PER_SERVER + local_gpu;
                void * dst_ptr = (char *) recvbuff + src_gpu_global_id * ncclTypeSize(datatype) * MAX_BUFFER_SIZE_PER_RANK + restore_recv_sched[local_rank_id * GPU_NUM_PER_SERVER + from_gpu].offset * ncclTypeSize(datatype);
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
    ret = ncclGroupEnd();
    if (ret == ncclInProgress) {
        do {
        ncclCommGetAsyncError(comm, &state);
        } while (state == ncclInProgress);
    } else if (ret == ncclSuccess) {
    /* Successfully issued */
    printf("Rank: %u, Kernel - step %u - issue succeeded\n", rankid, step_n-1);
    }

    return ncclSuccess;
}