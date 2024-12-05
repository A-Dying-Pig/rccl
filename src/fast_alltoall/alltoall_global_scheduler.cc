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
    gs->gpu_sched->gpu_n = _gpu_n;
    gs->gpu_sched->server_n = _server_n;
    gs->gpu_sched->rankid = rankid;

    gs -> buff_parameter = (buffer_parameter_t *) malloc(sizeof(buffer_parameter_t));
    memset(gs -> buff_parameter, 0 , sizeof(buffer_parameter_t));
}

void free_global_scheduler(struct GlobalScheduler * gs){
    free_matrix(&gs->mat);
     for (uint s = 0; s < gs->server_n; s++){
        free_local_scheduler(gs->locals[s]);
    }
    free(gs->sched);
    free(gs->gpu_sched);
    free(gs->buff_parameter);
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

    get_buffer_size(gs);

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
                            &gs->sched->steps[step_id].crossnode_sz[src_svr],
                            &gs->sched->steps[step_id + 1].restore[src_svr],
                            &gs->sched->steps[step_id + 1].restore_alltoall_sz[src_svr],
                            &gs->sched->steps[step_id + 1].direct_cpy[src_svr],
                            get_freq_permutation_set(&all2all.p_sets[pid]));
        }
        to_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].to_server);
        from_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].from_server);
        step_id ++;
    }
    gs->sched->step_n = all2all.p_sets_n + 1;


    schedule_this_gpu(gs);

    printf("------------Buffer size, rank: %u-----------------\n \
            sendbuff: %u\n, \
            recvbuff: %u\n, \
            lbsendbuff: %u\n, \
            lbrecvbuff: %u\n, \
            crosbuff: %u\n, \
            rstrbuff: %u\n \
            --------------------------------------------------\n",
            gs->gpu_sched->rankid,
            gs->buff_parameter->sendbuff_total_sz,
            gs->buff_parameter->recvbuff_total_sz,
            gs->buff_parameter->lbsend_total_sz,
            gs->buff_parameter->lbrecv_total_sz,
            gs->buff_parameter->crosbuff_total_sz,
            gs->buff_parameter->rstrbuff_total_sz);
}


void get_buffer_size(struct GlobalScheduler * gs){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;


    gs->buff_parameter->lbsend_total_sz = 0;
    gs->buff_parameter->lbrecv_total_sz = 0;
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){

        gs->buff_parameter->lbsend_disp[local_gpu] = gs->buff_parameter->lbsend_total_sz;
        gs->buff_parameter->lbrecv_disp[local_gpu] = gs->buff_parameter->lbrecv_total_sz;
        uint send_area_sz = 0, recv_area_sz = 0;
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){

            gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_disp[dst_gpu] = gs->buff_parameter->lbsend_total_sz;
            uint send_region_sz = 0, recv_region_sz = 0;
            bool send_lb = false, recv_lb = false;
            for (uint s = 0; s != server_n; s++){
                if (s == server_id){
                    continue;
                }
                size_t send_data_sz = (gs -> sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].sz[dst_gpu];
                if (send_data_sz > 0){
                    gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s] = gs->buff_parameter->lbsend_total_sz;
                    gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s] = send_data_sz;
                    gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_n ++;
                    gs->buff_parameter->lbsend_total_sz += send_data_sz;
                    send_region_sz += send_data_sz;
                    send_lb = true;
                }

                size_t recv_data_sz = (gs -> sched -> balance)[server_id][s][local_gpu * gpu_n + local_rank_id].sz[dst_gpu];
                if (recv_data_sz > 0){
                    gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s] = gs->buff_parameter->lbrecv_total_sz;
                    gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s] = recv_data_sz;
                    gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_n ++;
                    gs->buff_parameter->lbrecv_total_sz += recv_data_sz;
                    recv_region_sz += recv_data_sz;
                    recv_lb = true;
                }
            }
            gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_sz[dst_gpu] = send_region_sz;
            if (send_lb) gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_n ++;
            send_area_sz += send_region_sz;
            gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_sz[dst_gpu] = recv_region_sz;
            if (recv_lb) gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_n ++;
            recv_area_sz += recv_region_sz;

        }
        gs->buff_parameter->lbsend_sz[local_gpu] = send_area_sz;
        gs->buff_parameter->lbrecv_sz[local_gpu] = recv_area_sz;
    }


    gs->buff_parameter->sendbuff_total_sz = 0;
    for (uint i = 0; i < server_n * gpu_n; i ++){
        gs->buff_parameter->sendbuff_disp[i] = gs->buff_parameter->sendbuff_total_sz;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint lb_data_sz = gs->locals[server_id]->data_after_balance[local_rank_id][i].sz[src_gpu];
            if (lb_data_sz > 0){
                gs->buff_parameter->sendbuff_region[i].src_gpu_disp[src_gpu] = gs->buff_parameter->sendbuff_total_sz;
                gs->buff_parameter->sendbuff_region[i].src_gpu_sz[src_gpu] = lb_data_sz;
                gs->buff_parameter->sendbuff_total_sz += lb_data_sz;
                gs->buff_parameter->sendbuff_region[i].src_gpu_n ++;
            }
        }
    }

    gs->buff_parameter->recvbuff_total_sz = 0;
    for (uint i = 0; i < server_n; i ++){
        for (uint j = 0; j < gpu_n; j++){
            uint src_rank = i * gpu_n + j;
            gs->buff_parameter->recvbuff_disp[src_rank] = gs->buff_parameter->recvbuff_total_sz;
            gs->buff_parameter->recvbuff_sz[src_rank] = gs->locals[i]->data[j][global_rank_id];
            gs->buff_parameter->recvbuff_total_sz += gs->locals[i]->data[j][global_rank_id];
        }
    }

}


void schedule_this_gpu(struct GlobalScheduler * gs){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;

    // ------------------------------------------
    // Intrinsic alltoall: sendbuff => recvbuff
    //-------------------------------------------

    for (uint r = 0; r < gpu_n; r++){
        uint cur_gpu = server_id * gpu_n + r;
        size_t send_data_sz = gs -> buff_parameter -> sendbuff_sz[cur_gpu];
        if (send_data_sz > 0){
            uint send_data_disp = gs -> buff_parameter -> sendbuff_disp[cur_gpu];
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].gpu = cur_gpu;
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].disp = send_data_disp;
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].sz = send_data_sz;
            gs -> gpu_sched -> intrinsic_send_n ++;
        }

        size_t recv_data_sz = gs -> buff_parameter -> recvbuff_sz[cur_gpu];
        if (recv_data_sz > 0){
            uint recv_data_disp = gs -> buff_parameter -> recvbuff_disp[cur_gpu];
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].gpu = cur_gpu;
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].disp = recv_data_disp;
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].sz = recv_data_sz;
            gs -> gpu_sched -> intrinsic_recv_n ++;
        }
    }

    // --------------------------------------------------
    // Load balance:
    // First step: lbsend_buff ==> lbrecv_buff
    // Second step: lbrecv_buff ---(memcpy)--> sendbuff
    // --------------------------------------------------

    // first step
    for (uint r = 0; r < gpu_n; r++){
        size_t send_data_sz = gs -> buff_parameter -> lbsend_sz[r];
        uint cur_gpu_global_id = server_id * gpu_n + r;
        if (send_data_sz > 0){
            uint send_data_disp = gs -> buff_parameter -> lbsend_disp[r];
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].gpu = cur_gpu_global_id;
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].disp = send_data_disp;
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].sz = send_data_sz;
            gs -> gpu_sched -> balance_send_n ++;
        }
        size_t recv_data_sz = gs -> buff_parameter -> lbrecv_sz[r];
        if (recv_data_sz > 0){
            uint recv_data_disp = gs -> buff_parameter -> lbrecv_disp[r];
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].gpu = cur_gpu_global_id;
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].disp = recv_data_disp;
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].sz = recv_data_sz;
            gs -> gpu_sched -> balance_recv_n ++;
        }
    }
    // second step
    for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
            for (uint s = 0; s < server_n; s ++){
                if (s == server_id){
                    continue;
                }
                uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                uint cpy_sz = gs -> buff_parameter -> lbrecv_area[src_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                if (cpy_sz > 0){
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].src_disp =
                        gs -> buff_parameter -> lbrecv_area[src_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].dst_disp =
                        gs -> buff_parameter -> sendbuff_region[dst_gpu_global_id].src_gpu_disp[src_gpu];
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].sz = cpy_sz;
                    gs -> gpu_sched -> balance_memcpy_n ++;
                }
            }
        }
    }


    // ---------------------------------------------------------
    // Cross node: sendbuff ==> crosbuff
    // ---------------------------------------------------------

    //-----------------------------------------------------------
    // Data restore
    // First step: crosbuff ==> restorebuff
    // Second step: crosbuff -- (memcpy) -> recvbuff
    // Second step: restorebuff --(memcpy)-> recvbuff
    //-----------------------------------------------------------

    // calculate buffer size of crosbuff
    uint crosbuff_sz = 0;
    for (uint step_id = 1; step_id < gs->sched->step_n - 1; step_id++){
        crosbuff_sz = MAX(crosbuff_sz, (gs -> sched -> steps)[step_id].crossnode_sz[server_id][local_rank_id]);
    }
    // make it 512-byte aligned
    crosbuff_sz  = (crosbuff_sz + 0x1ff) & 0xfffffe00;
    gs -> buff_parameter->crosbuff_total_sz = crosbuff_sz * 2;
    gs -> buff_parameter->crosbuff_offset = crosbuff_sz;
    uint rstrbuff_sz = 0;


    // first step
    uint crosbuff_offset = gs -> buff_parameter -> crosbuff_offset;
    uint cur_offset = 0, prev_offset = 0;


    uint crossnode_send_disp[MAX_SERVER_NUM];
    memset(crossnode_send_disp, 0 , sizeof(uint) * MAX_SERVER_NUM);
    uint step_id = 0;
    gs -> gpu_sched -> step_n = gs->sched->step_n;
    struct scheduling_step_t * cur_step = &(gs -> sched -> steps)[0];
    struct scheduling_step_gpu_t * cur_gpu_step = &(gs -> gpu_sched -> steps)[0];
    uint dst_server = cur_step -> to_server[server_id];
    uint src_server = cur_step -> from_server[server_id];
    uint dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
    uint src_gpu_global_id = src_server * gpu_n + local_rank_id;

    (cur_gpu_step -> crossnode_send).sz = cur_step -> crossnode_sz[server_id][local_rank_id];
    (cur_gpu_step -> crossnode_send).gpu = dst_gpu_global_id;
    (cur_gpu_step -> crossnode_send).disp = gs -> buff_parameter -> sendbuff_disp[dst_server * gpu_n] + crossnode_send_disp[dst_server];
    crossnode_send_disp[dst_server] += (cur_gpu_step -> crossnode_send).sz;

    (cur_gpu_step -> crossnode_recv).sz = cur_step -> crossnode_sz[src_server][local_rank_id];
    (cur_gpu_step -> crossnode_recv).gpu = src_gpu_global_id;
    (cur_gpu_step -> crossnode_recv).disp = 0; // not applying offset here


    // middle steps
    uint prev_dst_server = dst_server,
        prev_src_server = src_server;
    struct scheduling_step_t * prev_step = cur_step;
    struct scheduling_step_gpu_t * prev_gpu_step = cur_gpu_step;
    struct recv_data_t * restore_send_sched, * restore_recv_sched, * dcopy_sched;
    uint restore_alltoall_senddisp = 0, restore_alltoall_recvdisp = 0, direct_cpy_disp = 0, restore_recvdisp[MAX_GPU_PER_SERVER], direct_cpy_src_disp = 0;

    for (step_id = 1; step_id < gs->sched->step_n - 1; step_id++){
        cur_step = &(gs -> sched -> steps)[step_id];
        cur_gpu_step =  &(gs -> gpu_sched -> steps)[step_id];
        dst_server = cur_step -> to_server[server_id];
        src_server = cur_step -> from_server[server_id];

        dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
        src_gpu_global_id = src_server * gpu_n + local_rank_id;

        cur_offset = (step_id % 2 == 1) ? crosbuff_offset : 0;
        prev_offset = ((step_id - 1) % 2 == 1) ? 0 : crosbuff_offset;
        // cross node transfer

        (cur_gpu_step -> crossnode_send).sz = cur_step -> crossnode_sz[server_id][local_rank_id];
        (cur_gpu_step -> crossnode_send).gpu = dst_gpu_global_id;
        (cur_gpu_step -> crossnode_send).disp = gs -> buff_parameter -> sendbuff_disp[dst_server * gpu_n] + crossnode_send_disp[dst_server];
        crossnode_send_disp[dst_server] += (cur_gpu_step -> crossnode_send).sz;

        (cur_gpu_step -> crossnode_recv).sz = cur_step -> crossnode_sz[src_server][local_rank_id];
        (cur_gpu_step -> crossnode_recv).gpu = src_gpu_global_id;
        (cur_gpu_step -> crossnode_recv).disp = cur_offset; // applying offset here

        //restore data from previous step

        // restore alltoall
        restore_alltoall_senddisp = 0;
        restore_alltoall_recvdisp = 0;
        direct_cpy_disp = 0;
        memset(restore_recvdisp, 0, sizeof(uint) * MAX_GPU_PER_SERVER);
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            size_t send_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_rank_id][local_gpu];
            if (local_gpu == local_rank_id){
                direct_cpy_disp = restore_alltoall_senddisp;
                restore_alltoall_senddisp += send_data_sz;
                continue;
            }

            uint cur_gpu = server_id * gpu_n + local_gpu;
            if (send_data_sz > 0){
                cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].gpu = cur_gpu;
                cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].disp = restore_alltoall_senddisp + prev_offset;    //applying offset
                cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].sz = send_data_sz;
                restore_alltoall_senddisp += send_data_sz;
                cur_gpu_step -> restore_send_n ++;
            }
            size_t recv_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_gpu][local_rank_id];
            if (recv_data_sz > 0){
                cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].gpu = cur_gpu;
                cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].disp = restore_alltoall_recvdisp;
                restore_recvdisp[local_gpu] = restore_alltoall_recvdisp;
                cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].sz = recv_data_sz;
                restore_alltoall_recvdisp += recv_data_sz;
                cur_gpu_step -> restore_recv_n ++;
                // calculate restore buffer size
                rstrbuff_sz = MAX(rstrbuff_sz, recv_data_sz);
            }
        }

        // direct copy
        direct_cpy_src_disp = 0;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint cpy_sz = cur_step->direct_cpy[prev_src_server][local_rank_id][src_gpu].sz;
            if (cpy_sz > 0){
                uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
                cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].sz = cpy_sz;
                cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].src_disp = direct_cpy_disp + direct_cpy_src_disp + prev_offset; // applying offset
                cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> direct_cpy[prev_src_server][local_rank_id][src_gpu].offset ;
                direct_cpy_src_disp += cpy_sz;
                cur_gpu_step -> direct_memcpy_n ++;
            }
        }

        // restore memcpy
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            uint restore_cpy_src_disp = 0;
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
                uint cpy_sz = cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].sz;
                if(cpy_sz > 0){
                    uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
                    cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].sz = cpy_sz;
                    cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].src_disp = restore_recvdisp[local_gpu] + restore_cpy_src_disp;
                    cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].offset;
                    restore_cpy_src_disp += cpy_sz;
                    cur_gpu_step -> restore_memcpy_n ++;
                }
            }
        }

        prev_src_server = src_server;
        prev_dst_server = dst_server;
    }

    // final restore
    prev_offset = ((gs -> sched -> step_n - 1) % 2 == 1) ? 0 : crosbuff_offset;
    cur_step = &(gs -> sched -> steps)[ gs -> sched -> step_n - 1];
    cur_gpu_step = &(gs -> gpu_sched -> steps)[ gs -> gpu_sched -> step_n - 1];

    // restore alltoall
    restore_alltoall_senddisp = 0;
    restore_alltoall_recvdisp = 0;
    direct_cpy_disp = 0;
    memset(restore_recvdisp, 0, sizeof(uint) * MAX_GPU_PER_SERVER);
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
        size_t send_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_rank_id][local_gpu];
        if (local_gpu == local_rank_id){
            direct_cpy_disp = restore_alltoall_senddisp;
            restore_alltoall_senddisp += send_data_sz;
            continue;
        }

        uint cur_gpu = server_id * gpu_n + local_gpu;
        if (send_data_sz > 0){
            cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].gpu = cur_gpu;
            cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].disp = restore_alltoall_senddisp + prev_offset;    // applying offset
            cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].sz = send_data_sz;
            restore_alltoall_senddisp += send_data_sz;
            cur_gpu_step -> restore_send_n ++;
        }
        size_t recv_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_gpu][local_rank_id];
        if (recv_data_sz > 0){
            cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].gpu = cur_gpu;
            cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].disp = restore_alltoall_recvdisp;
            restore_recvdisp[local_gpu] = restore_alltoall_recvdisp;
            cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].sz = recv_data_sz;
            restore_alltoall_recvdisp += recv_data_sz;
            cur_gpu_step -> restore_recv_n ++;
            // calculate restore buffer size
            rstrbuff_sz = MAX(rstrbuff_sz, recv_data_sz);
        }
    }

    // direct copy
    direct_cpy_src_disp = 0;
    for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
        uint cpy_sz = cur_step->direct_cpy[prev_src_server][local_rank_id][src_gpu].sz;
        if (cpy_sz > 0){
            uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
            cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].sz = cpy_sz;
            cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].src_disp = direct_cpy_disp + direct_cpy_src_disp+ prev_offset; // applying offset
            cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> direct_cpy[prev_src_server][local_rank_id][src_gpu].offset ;
            direct_cpy_src_disp += cpy_sz;
            cur_gpu_step -> direct_memcpy_n ++;
        }
    }

    // restore memcpy
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
        uint restore_cpy_src_disp = 0;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint cpy_sz = cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].sz;
            if(cpy_sz > 0){
                uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
                cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].sz = cpy_sz;
                cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].src_disp = restore_recvdisp[local_gpu] + restore_cpy_src_disp;
                cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].offset;
                restore_cpy_src_disp += cpy_sz;
                cur_gpu_step -> restore_memcpy_n ++;
            }
        }
    }

    gs -> buff_parameter -> rstrbuff_total_sz = rstrbuff_sz;
}
