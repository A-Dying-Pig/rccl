#pragma once
#include <iostream>
#include "alltoall_matrix.h"
#include "alltoall_local_scheduler.h"
#include "alltoall_algorithm.h"
#include "alltoall_define.h"



struct scheduling_step_t{
    uint to_server[MAX_GPU_PER_SERVER];
    uint from_server[MAX_GPU_PER_SERVER];
    // ChannelPtr: gpu_n * gpu_n (row -> remote dst_gpu's local id, col -> from gpu)
    uint channel[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE];
    //  RestorePtr: gpu_n * gpu_n (row -> dst_gpu's local id, col -> from gpu)
    struct recv_data_t restore[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE];
    // server id * channel id
    struct recv_data_t direct_cpy[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE];
};

struct scheduling_result_t{
    uint gpu_n;
    uint server_n;
    uint rankid;
    uint MAX_BUFFER_SIZE_PER_RANK;
    struct balance_data_t balance[MAX_SERVER_NUM][MAX_SERVER_NUM][MAX_GPU_PER_SERVER_SQUARE];
    struct scheduling_step_t steps[MAX_TRANSFER_STEP_NUM];
    uint step_n;
    uint intrinsic_ata[MAX_SERVER_NUM][MAX_GPU_PER_SERVER_SQUARE];
};




struct memcopy_buffer_t{
    uint src_disp;
    uint dst_disp;
    uint sz;
};

struct send_recv_buffer_t{
    uint gpu;
    uint disp;
    uint sz;
};

struct scheduling_step_gpu_t{
    struct send_recv_buffer_t crossnode_send;
    struct send_recv_buffer_t crossnode_recv;
    struct send_recv_buffer_t restore[GPU_NUM_PER_SERVER];
    uint restore_n;
    struct memcopy_buffer_t restore_memcpy[GPU_NUM_PER_SERVER];    // including direct copy
    uint restore_memcpy_n;
};

// scheduling result for a particular gpu
struct scheduling_result_gpu_t{
    uint gpu_n;
    uint server_n;
    uint rankid;
    uint sendbuff_sz;
    uint restore
    // intrinsic alltoall metadata
    struct send_recv_buffer_t intrinsic_send[GPU_NUM_PER_SERVER];
    uint intrinsic_send_n;
    struct send_recv_buffer_t intrinsic_recv[GPU_NUM_PER_SERVER];
    uint intrinsic_recv_n;
    // load balance metadata
    struct send_recv_buffer_t balance_send[GPU_NUM_PER_SERVER];
    uint balance_send_n;
    struct send_recv_buffer_t balance_recv[GPU_NUM_PER_SERVER];
    uint balance_send_n;

    struct memcopy_buffer_t balance_memcpy[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE];
    uint balance_memcpy_n;

    struct scheduling_step_gpu_t steps[MAX_TRANSFER_STEP_NUM];
    uint step_n;
};

struct GlobalScheduler{
    uint server_n;
    uint gpu_n;
    struct Matrix mat;
    struct LocalScheduler * locals[MAX_SERVER_NUM];
    struct scheduling_result_t * sched;
    struct scheduling_result_gpu_t * gpu_sched;
};


void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint * demand_matrix, uint rankid);
void free_global_scheduler(struct GlobalScheduler * gs);
void run_scheduler(struct GlobalScheduler * gs);
void arrange_buffers(struct GlobalScheduler * gs);




struct alltoall_buffer{
    void * sendbuff;
    uint sendbuff_sz;
    void * recvbuff;
    uint recvbuff_sz;
    void * crosbuff;
    uint crosbuff_sz;
    void * rstrbuff;
    uint rstrbuff_sz;
};

struct alltoall_buffer allocate_buffer(struct GlobalScheduler * gs);



