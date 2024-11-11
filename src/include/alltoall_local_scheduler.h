#pragma once

#include <iostream>
#include "alltoall_define.h"
#include <map>

using namespace std;


#define MAX_GPU_PER_SERVER 16

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

class LocalScheduler{
private:
    uint ** data;
    uint ** balanced_data;
    struct data_t ** data_after_balance; 
    uint * intrinsic_all2all;
    uint gpu_n;
    uint server_n;
    uint server_id;
    uint * row_sum; //row sum at each tile; gpu_n * server_n

public:
    uint * server2server_data;
    LocalScheduler(uint* _data, uint _gpu_n, uint _server_n, uint _server_id);
    ~LocalScheduler();
    void prepare_load_balance();
    // void balance_one_server(uint to_server_id, BalancePtr r);   // r is a transfer matrix before this server talks to another server to balance data
    void balance_one_server2(uint to_server_id, BalancePtr r);   // r is a transfer matrix before this server talks to another server to balance data
    // void restore_one_server(uint to_server_id, RestorePtr r, DirectCpyPtr dcpy, uint freq);    // r is a transfer matrix after this server talks to another server to restore data
    // ChannelPtr is for transfer from one channel to another, gpu_n * gpu_n matrix, row represents dst_gpu_id , col represnets from_gpu_id
    // RestorePtr is for transfer from one channel to another
    void restore_one_server2(uint to_server_id, vector<ChannelPtr> channel, vector<RestorePtr> r, vector<DirectCpyPtr> dcpy, uint freq);    // r is a transfer matrix after this server talks to another server to restore data
    uint get_server_id(){return server_id;}
    uint * get_intrinsic_all2all(){return intrinsic_all2all;}
    void print();
    void print(uint dst_server_id);
};

void print_matrix(uint * data, uint m, uint n);
