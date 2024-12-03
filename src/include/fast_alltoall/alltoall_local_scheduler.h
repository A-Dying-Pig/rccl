#pragma once

#include "alltoall_define.h"

struct data_t{
    uint sz[MAX_GPU_PER_SERVER];
    uint offset[MAX_GPU_PER_SERVER];
    uint sum;
};

struct recv_data_t{
    uint sz;
    uint offset;
};

struct balance_data_t{
    uint sz[MAX_GPU_PER_SERVER];    // transferred data size for each local gpu
};

struct LocalScheduler{
    uint ** data;
    uint ** balanced_data;
    struct data_t ** data_after_balance;
    uint * intrinsic_all2all;
    uint gpu_n;
    uint server_n;
    uint server_id;
    uint * row_sum; //row sum at each tile; gpu_n * server_n
    uint * server2server_data;
};

void init_local_scheduler(struct LocalScheduler * ls, uint* _data, uint _gpu_n, uint _server_n, uint _server_id);
void free_local_scheduler(struct LocalScheduler * ls);
void prepare_load_balance(struct LocalScheduler * ls);
void balance_one_server(struct LocalScheduler * ls, uint to_server_id, struct balance_data_t (*r)[MAX_GPU_PER_SERVER_SQUARE]);   // r is a transfer matrix before this server talks to another server to balance data
void restore_one_server(struct LocalScheduler * ls, uint to_server_id, uint (*channel)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], struct recv_data_t (*r)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], struct recv_data_t (*dcpy)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], uint freq);    // r is a transfer matrix after this server talks to another server to restore data
void print_local_scheduler(struct LocalScheduler * ls);
void print_local_scheduler(struct LocalScheduler * ls, uint dst_server_id);

void print_matrix(uint * data, uint m, uint n);


void intrinsic_alltoall_gpu(
    uint rankid,
    struct LocalScheduler * ls,
    uint (*send)[GPU_NUM_PER_SERVER],
    uint * send_n,
    uint (*recv)[GPU_NUM_PER_SERVER],
    uint recv_n
);

void balance_servers_gpu(
    uint rankid,
    struct LocalScheduler * ls,
    struct balance_data_t (*send)[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER],
    uint * send_n,
    struct balance_data_t (*recv)[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER],
    uint * recv_n);

void restore_one_server_gpu(
    uint rankid,
    struct LocalScheduler * ls,
    uint to_server_id,
    uint crossnode,
    struct recv_data_t (*restore_send)[GPU_NUM_PER_SERVER],
    uint * restore_send_n,
    struct recv_data_t (*restore_recv)[GPU_NUM_PER_SERVER],
    uint * restore_recv_n,
    uint (*cpy)[GPU_NUM_PER_SERVER],
    uint * cpy_n,
    uint freq);