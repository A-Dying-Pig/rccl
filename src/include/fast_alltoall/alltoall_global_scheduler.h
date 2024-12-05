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
    uint crossnode_sz[MAX_SERVER_NUM][MAX_GPU_PER_SERVER];
    uint restore_alltoall_sz[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER];
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


struct memcpy_buffer_t{
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
    struct send_recv_buffer_t restore_send[GPU_NUM_PER_SERVER];
    uint restore_send_n;
    struct send_recv_buffer_t restore_recv[GPU_NUM_PER_SERVER];
    uint restore_recv_n;
    struct memcpy_buffer_t direct_memcpy[GPU_NUM_PER_SERVER];
    uint direct_memcpy_n;
    struct memcpy_buffer_t restore_memcpy[MAX_SERVER_NUM_SQUARE];
    uint restore_memcpy_n;
};

// scheduling result for a particular gpu
struct scheduling_result_gpu_t{
    uint gpu_n;
    uint server_n;
    uint rankid;
    // intrinsic alltoall metadata
    struct send_recv_buffer_t intrinsic_send[GPU_NUM_PER_SERVER];
    uint intrinsic_send_n;
    struct send_recv_buffer_t intrinsic_recv[GPU_NUM_PER_SERVER];
    uint intrinsic_recv_n;
    // load balance metadata
    struct send_recv_buffer_t balance_send[GPU_NUM_PER_SERVER];
    uint balance_send_n;
    struct send_recv_buffer_t balance_recv[GPU_NUM_PER_SERVER];
    uint balance_recv_n;

    struct memcpy_buffer_t balance_memcpy[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE];
    uint balance_memcpy_n;

    struct scheduling_step_gpu_t steps[MAX_TRANSFER_STEP_NUM];
    uint step_n;
};


struct sendbuff_region_t{
    uint src_gpu_disp[GPU_NUM_PER_SERVER];
    uint src_gpu_sz[GPU_NUM_PER_SERVER];
    uint src_gpu_n;
};


struct lbbuff_region_t{
    uint server_disp[MAX_SERVER_NUM];
    uint server_sz[MAX_SERVER_NUM];
    uint server_n;
};

struct lbbuff_area_t{
    struct lbbuff_region_t dst_gpu_region[GPU_NUM_PER_SERVER];
    uint dst_gpu_disp[GPU_NUM_PER_SERVER];
    uint dst_gpu_sz[GPU_NUM_PER_SERVER];
    uint dst_gpu_n;
};

struct buffer_parameter_t{
    uint sendbuff_disp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint sendbuff_sz[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    struct sendbuff_region_t sendbuff_region[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint sendbuff_total_sz;

    uint recvbuff_disp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint recvbuff_sz[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint recvbuff_total_sz;


    uint lbsend_disp[GPU_NUM_PER_SERVER];
    uint lbsend_sz[GPU_NUM_PER_SERVER];
    struct lbbuff_area_t lbsend_area[GPU_NUM_PER_SERVER];
    uint lbsend_total_sz;

    uint lbrecv_disp[GPU_NUM_PER_SERVER];
    uint lbrecv_sz[GPU_NUM_PER_SERVER];
    struct lbbuff_area_t lbrecv_area[GPU_NUM_PER_SERVER];
    uint lbrecv_total_sz;

    uint crosbuff_total_sz; // use the offset to alternate the first and second half of the buffer
    uint crosbuff_offset;
    uint rstrbuff_total_sz;
};

struct GlobalScheduler{
    uint server_n;
    uint gpu_n;
    struct Matrix mat;
    struct LocalScheduler * locals[MAX_SERVER_NUM];
    struct scheduling_result_t * sched;
    struct scheduling_result_gpu_t * gpu_sched;
    struct buffer_parameter_t * buff_parameter;
};


void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint * demand_matrix, uint rankid);
void free_global_scheduler(struct GlobalScheduler * gs);
void run_scheduler(struct GlobalScheduler * gs);
void get_buffer_size(struct GlobalScheduler * gs);
void schedule_this_gpu(struct GlobalScheduler * gs);


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




