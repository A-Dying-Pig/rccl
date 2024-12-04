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
        memcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(uint));
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
    uint lbrecv_total_sz

};


void get_buffer_size(struct GlobalScheduler * gs, struct buffer_parameter_t * buff_parameter){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;


    buff_parameter->lbsend_total_sz = 0;
    buff_parameter->lbrecv_total_sz = 0;
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){

        buff_parameter->lbsend_disp[local_gpu] = buff_parameter->lbsend_total_sz;
        buff_parameter->lbrecv_disp[local_gpu] = buff_parameter->lbrecv_total_sz;
        uint send_area_sz = 0, recv_area_sz = 0;
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){

            buff_parameter->lbsend_area[local_gpu].dst_gpu_disp[dst_gpu] = buff_parameter->lbsend_total_sz;
            uint send_region_sz = 0, recv_region_sz = 0;
            bool send_lb = false, recv_lb = false;
            for (uint s = 0; s != server_n; s++){
                if (s == server_id){
                    continue;
                }
                size_t send_data_sz = (gs -> sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].sz[dst_gpu];
                if (send_data_sz > 0){
                    buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s] = buff_parameter->lbsend_total_sz;
                    buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s] = send_data_sz;
                    buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_n ++;
                    buff_parameter->lbsend_total_sz += send_data_sz;
                    send_region_sz += send_data_sz;
                    send_lb = true;
                }



                size_t recv_data_sz = (gs -> sched -> balance)[server_id][s][local_gpu * gpu_n + local_rank_id].sz[dst_gpu];
                if (recv_data_sz > 0){
                    buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s] = buff_parameter->lbrecv_total_sz;
                    buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s] = recv_data_sz;
                    buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_n ++;
                    buff_parameter->lbrecv_total_sz += recv_data_sz;
                    recv_region_sz += recv_data_sz;
                    recv_lb = true;
                }
            }
            buff_parameter->lbsend_area[local_gpu].dst_gpu_sz[dst_gpu] = send_region_sz;
            if (send_lb) buff_parameter->lbsend_area[local_gpu].dst_gpu_n ++;
            send_area_sz += send_region_sz;
            buff_parameter->lbrecv_area[local_gpu].dst_gpu_sz[dst_gpu] = recv_region_sz;
            if (recv_lb) buff_parameter->lbrecv_area[local_gpu].dst_gpu_n ++;
            recv_area_sz += recv_region_sz;

        }
        buff_parameter->lbsend_sz[local_gpu] = send_area_sz;
        buff_parameter->lbrecv_sz[local_gpu] = recv_area_sz;
    }


    buff_parameter->sendbuff_total_sz = 0;
    for (uint i = 0; i < server_n * gpu_n; i ++){
        buff_parameter->sendbuff_disp[i] = buff_parameter->sendbuff_total_sz;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint lb_data_sz = gs->locals[server_id]->data_after_balance[local_rank_id][i].sz[src_gpu];
            if (lb_data_sz > 0){
                buff_parameter->sendbuff_region[i].src_gpu_disp[src_gpu] = buff_parameter->sendbuff_total_sz;
                buff_parameter->sendbuff_region[i].src_gpu_sz[src_gpu] = lb_data_sz;
                buff_parameter->sendbuff_total_sz += lb_data_sz;
                sendbuff_region[i].src_gpu_n ++;
            }
        }
    }

    buff_parameter->recvbuff_total_sz = 0;
    for (uint i = 0; i < server_n; i ++){
        for (uint j = 0; j < gpu_n; j++){
            uint src_rank = i * gpu + j;
            buff_parameter->recvbuff_disp[src_rank] = buff_parameter->recvbuff_total_sz;
            buff_parameter->recvbuff_sz[src_rank] = gs->locals[i]->data[j][global_rank_id];
            buff_parameter->recvbuff_total_sz += gs->locals[i]->data[j][global_rank_id];
        }
    }

}


void arrange_buffers(struct GlobalScheduler * gs, struct buffer_parameter_t * buff_parameter){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;





}
