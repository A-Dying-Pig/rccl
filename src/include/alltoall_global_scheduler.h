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
    uint channel[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER];
    //  RestorePtr: gpu_n * gpu_n (row -> dst_gpu's local id, col -> from gpu)
    struct recv_data_t restore[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER];
    // server id * channel id
    struct recv_data_t direct_cpy[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER];
};

struct scheduling_result_t{
    BalancePtr balance[MAX_SERVER_NUM][MAX_SERVER_NUM];
    struct scheduling_step_t steps[MAX_SERVER_NUM_SQUARE];
    uint step_n;
    TransferMatrixElement* intrinsic_ata[MAX_SERVER_NUM];
};


struct GlobalScheduler{
    uint server_n;
    uint gpu_n;
    struct Matrix mat;
    struct LocalScheduler * locals[MAX_SERVER_NUM];
    struct scheduling_result_t * sched;
};


void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint * demand_matrix);
void free_global_scheduler(struct GlobalScheduler * gs);
void run_scheduler(struct GlobalScheduler * gs);




