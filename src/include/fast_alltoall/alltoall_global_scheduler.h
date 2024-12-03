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


struct scheduling_step_gpu_t{
    uint to_server;
    uint from_server;
    uint crossnode;
    struct recv_data_t restore_send[GPU_NUM_PER_SERVER];
    uint restore_send_n;
    struct recv_data_t restore_recv[GPU_NUM_PER_SERVER];
    uint restore_recv_n;
    uint cpy[GPU_NUM_PER_SERVER];
    uint cpy_n;
};

// scheduling result for a particular gpu
struct scheduling_result_gpu_t{
    uint gpu_n;
    uint server_n;
    uint rankid;
    uint sendbuff_sz;
    uint restore
    uint intrinsic_send[GPU_NUM_PER_SERVER];
    uint intrinsic_send_n;
    uint intrinsic_recv[GPU_NUM_PER_SERVER];
    uint intrinsic_recv_n;
    struct balance_data_t balance_send[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER]; // dst server x send_gpu
    uint balance_send_n;
    struct balance_data_t balance_recv[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER]; // dst server x recv_gpu
    uint balance_recv_n;
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
void run_scheduler_gpu(struct GlobalScheduler * gs);




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



