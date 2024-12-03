#include "fast_alltoall/alltoall_global_scheduler.h"
#include "fast_alltoall/alltoall_define.h"
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>


void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint * demand_matrix,  uint rankid){
    gs->server_n = _server_n;
    gs->gpu_n = _gpu_n;
    uint dim = gs->gpu_n * gs->server_n;
    for (uint s = 0; s < gs->server_n; s++){
        gs->locals[s] = (LocalScheduler *) malloc(sizeof(LocalScheduler));
        // hipMallocManaged((void**)&gs->locals[s], sizeof(LocalScheduler));
        init_local_scheduler(gs->locals[s], demand_matrix + s * dim * gs->gpu_n, gs->gpu_n, gs->server_n, s);
    }
    uint * data = (uint *) malloc(sizeof(uint) * gs->server_n * gs->server_n);
    // hipMallocManaged((void**)&data, sizeof(uint) * gs->server_n * gs->server_n);
    // uint *data = new uint[server_n * server_n];
    for (uint s = 0; s < gs->server_n; s++){
       uint src_svr = gs->locals[s] -> server_id;
        for (uint j = 0; j < gs->server_n; j++){
            data[src_svr * gs->server_n + j] =  gs->locals[s]->server2server_data[j];
       }
    }
    init_matrix(&gs->mat);
    copy_matrix(&gs->mat, data, gs->server_n);
    free(data);
    // hipFree((void*) data);
    gs->sched = (scheduling_result_t *) malloc(sizeof(scheduling_result_t));
    memset(gs->sched, 0, sizeof(scheduling_result_t));
    // hipMallocManaged((void**) &gs->sched, sizeof(scheduling_result_t));
    // hipMemset(gs->sched, 0, sizeof(scheduling_result_t));
    gs->sched->gpu_n = _gpu_n;
    gs->sched->server_n = _server_n;

    gs->gpu_sched = (scheduling_result_gpu_t *) malloc(sizeof(scheduling_result_gpu_t));
    memset(gs->gpu_sched, 0, sizeof(scheduling_result_gpu_t));
    gs->gpu_sched->gpu_n = gpu_n;
    gs->gpu_sched->server_n = server_n;
    gs->gpu_sched->rankid = rankid;
}

void free_global_scheduler(struct GlobalScheduler * gs){
    free_matrix(&gs->mat);
     for (uint s = 0; s < gs->server_n; s++){
        free_local_scheduler(gs->locals[s]);
    }
    free(gs->sched);
    free(gs->gpu_sched);
    // hipFree(gs->sched);
}


void run_scheduler(struct GlobalScheduler * gs){
    FastAll2All all2all;
    init_fastall2all(&all2all, &gs->mat);
    to_scaled_doubly_stochastic_matrix_fastall2all(&all2all);
    decompose_fastall2all(&all2all);
    LOG("verify deccomposition: %u\n", verify_decomposition_fastall2all(&all2all));
    uint pid = 0, lid = 0;

    /* Start Pipelining*/

    // generate schedule for intra-server all2all - balance first
    // balance once

    for (lid = 0; lid < gs->server_n; lid++){
        for (uint s = 0; s < gs->server_n; s++){
           if (s == gs->locals[lid]->server_id){
            continue;
           }
           uint src_svr = gs->locals[lid]->server_id;
           balance_one_server(gs->locals[lid], s, &gs->sched->balance[src_svr][s]);
        }
    }

    // get intrinsic all-to-all
    for (lid = 0; lid < gs->server_n; lid++){
        uint src_svr = gs->locals[lid]->server_id;
        memcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(TransferMatrixElement));
        // hipMemcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(TransferMatrixElement), hipMemcpyHostToHost);
    }

    uint step_id = 0;
    for (pid = 0; pid < all2all.p_sets_n; pid++){
        for (lid = 0; lid < gs->server_n; lid++){
            uint src_svr = gs->locals[lid]->server_id;
            uint dst_svr = 0;
            map_lookup(all2all.p_sets[pid].mp, all2all.p_sets[pid].mp_n, src_svr, &dst_svr);
            restore_one_server(gs->locals[lid],
                            dst_svr, &gs->sched->steps[step_id].channel[src_svr],
                            &gs->sched->steps[step_id + 1].restore[src_svr],
                            &gs->sched->steps[step_id + 1].direct_cpy[src_svr],
                            get_freq_permutation_set(&all2all.p_sets[pid]));
        }
        to_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].to_server);
        from_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].from_server);
        step_id ++;
    }
    gs->sched->step_n = all2all.p_sets_n + 1;
}


void run_scheduler_gpu(struct GlobalScheduler * gs){



}


struct alltoall_buffer allocate_buffer(struct GlobalScheduler * gs){
    //----------------------------------------------------------------------
    // Send buff: | DST RANK 0 | DST RANK 1 | DST RANK 2 | ... | DST RANK n |
    //  ----Each DST RANK are sharded into at most gpu# regions, region i stores data whose source is local GPU i




}