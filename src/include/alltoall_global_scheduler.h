#pragma once
#include <iostream>
#include <vector>
#include "alltoall_matrix.h"
#include "alltoall_local_scheduler.h"
#include "alltoall_algorithm.h"
#include "alltoall_define.h"

using namespace std;


struct scheduling_step_t{
    vector<uint> to_server;
    vector<uint> from_server; 
    vector<vector<ChannelPtr> > channel; // ChannelPtr: gpu_n * gpu_n (row -> remote dst_gpu's local id, col -> from gpu)
    vector<vector<RestorePtr> > restore; //  RestorePtr: gpu_n * gpu_n (row -> dst_gpu's local id, col -> from gpu)
    vector<vector<DirectCpyPtr> > direct_cpy; // server id * channel id
};

struct scheduling_result_t{
    vector<vector<BalancePtr> > balance;
    vector<struct scheduling_step_t> steps;
    vector<TransferMatrixElement*> intrinsic_ata;
};


class GlobalScheduler{
private:
    uint server_n;
    uint gpu_n;
    Matrix mat;
    vector<LocalScheduler*> locals; 
public:
    GlobalScheduler(uint _server_n, uint _gpu_n, vector<LocalScheduler*> _locals);
    GlobalScheduler(uint _server_n, uint _gpu_n, uint * demand_matrix);
    ~GlobalScheduler();
    struct scheduling_result_t run();
};



