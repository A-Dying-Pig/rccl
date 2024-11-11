#include "alltoall_global_scheduler.h"
#include "alltoall_define.h"
#include <chrono>
#include <iostream>
using namespace chrono;
using namespace std;

GlobalScheduler::GlobalScheduler(uint _server_n, uint _gpu_n, vector<LocalScheduler*> _locals){
    server_n = _server_n;
    gpu_n = _gpu_n;
    locals = _locals;
    // contruct the matrix from the local scheduler result
    uint *data = new uint[server_n * server_n];
    for (auto local = locals.begin(); local != locals.end(); local++){
       uint src_svr = (*local) -> get_server_id();
       for (uint j = 0; j < server_n; j++){
            data[src_svr * server_n + j] = (*local)->server2server_data[j];
       }
    }
    mat.copy(data, server_n);
    // cout << "Global scheduler prints server2server matrix: " << endl;
    // mat.print();
    delete[] data;
}

GlobalScheduler::GlobalScheduler(uint _server_n, uint _gpu_n, uint * demand_matrix){
    server_n = _server_n;
    gpu_n = _gpu_n;
    uint dim = gpu_n * server_n;
    for (uint s = 0; s < server_n; s++){
        LocalScheduler* ls = new LocalScheduler(demand_matrix + s * dim * gpu_n, gpu_n, server_n, s);
        locals.push_back(ls);
    }
    uint *data = new uint[server_n * server_n];
    for (auto local = locals.begin(); local != locals.end(); local++){
       uint src_svr = (*local) -> get_server_id();
       for (uint j = 0; j < server_n; j++){
            data[src_svr * server_n + j] = (*local)->server2server_data[j];
            // cout << "server id : " << src_svr <<" to server " << j <<" data : " <<  (*local)->server2server_data[j] <<endl;
       }
    }
    // print_matrix(data, server_n, server_n);
    mat.copy(data, server_n);
    // cout << "Global scheduler prints server2server matrix: " << endl;
    // mat.print();
    delete[] data;

}

GlobalScheduler::~GlobalScheduler(){
    for (auto it = locals.begin(); it != locals.end(); it++){
        delete *it;
    }
}

struct scheduling_result_t GlobalScheduler::run(){
    FastAll2All all2all(&mat, gpu_n);
    all2all.to_scaled_doubly_stochastic_matrix();
    all2all.decompose();
    cout << "verify deccomposition: " << all2all.verify_decomposition() << endl;

    // cout << "birkhoff decompostion succeed, pset_n:  " << all2all.p_sets.size() << " ,MBpu: " << MBpu <<endl;
    // //print first result
    // all2all.p_sets[0].print_permutation_matrix();
    // cout << "frequency: " << all2all.p_sets[0].get_freq() << endl;

    // load balance in the begining (i.e., convert to balanced workload), restore data at each step
    uint locals_sz = locals.size(), pset_sz = all2all.p_sets.size();
    uint pid = 0;
    uint lid = 0;

    /* Start Pipelining*/
    struct scheduling_result_t schd_ret;
    
    // generate schedule for intra-server all2all - balance first
    // balance once
    vector <vector<BalancePtr> > bs(locals_sz, vector<BalancePtr>(server_n, NULL));
    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < server_n; s++){
            bs[lid][s] = new balance_data_t[gpu_n * gpu_n];
            memset(bs[lid][s], 0, sizeof(balance_data_t) * gpu_n * gpu_n);
        }
    }

    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < locals_sz; s++){
           if (s == locals[lid]->get_server_id()){
            continue;
           }
           locals[lid] -> balance_one_server2(s, bs[lid][s]);
    //     cout << lid * server_n + s << endl;
        }
    }
    schd_ret.balance = bs;

    // get intrinsic all-to-all
    for (uint i = 0; i < locals_sz; i++){
        TransferMatrixElement * intra_ata = new TransferMatrixElement[gpu_n * gpu_n];
        memcpy(intra_ata, locals[i]->get_intrinsic_all2all(), gpu_n * gpu_n * sizeof(TransferMatrixElement));
        schd_ret.intrinsic_ata.push_back(intra_ata);
    }

    // get transfer steps
    vector <vector<vector<ChannelPtr> > > chnl(pset_sz, vector<vector<ChannelPtr> >(locals_sz, vector<ChannelPtr>(gpu_n, NULL)));
    vector <vector<vector<RestorePtr> > > ds(pset_sz, vector<vector<RestorePtr> >(locals_sz, vector<RestorePtr>(gpu_n, NULL)));
    vector <vector<vector<DirectCpyPtr> > > d_cpy(pset_sz, vector<vector<DirectCpyPtr> >(locals_sz, vector<DirectCpyPtr>(gpu_n, NULL)));
    // model transfer and restore for each pset
    for (pid = 0; pid != pset_sz; pid++){
        for (lid = 0; lid != locals_sz; lid ++){
            for (uint cid = 0; cid != gpu_n; cid++){
                ds[pid][lid][cid] = new recv_data_t[gpu_n * gpu_n];
                memset(ds[pid][lid][cid], 0,  sizeof(recv_data_t) * gpu_n * gpu_n);
                d_cpy[pid][lid][cid] = new recv_data_t[gpu_n];
                memset(d_cpy[pid][lid][cid], 0,  sizeof(recv_data_t) * gpu_n);
                chnl[pid][lid][cid] = new TransferMatrixElement[gpu_n * gpu_n];
                memset(chnl[pid][lid][cid], 0,  sizeof(TransferMatrixElement) * gpu_n * gpu_n);
            }
           
        }
    }

    // generate schedule for intra-server all2all - restore for each step
    pid = 0;
    // cout << "pset number: " << all2all.p_sets.size() << endl;
    for (auto pset = all2all.p_sets.begin(); pset != all2all.p_sets.end(); pset++){
        lid = 0;
        // cout << "pset freq: " << pset -> get_freq() << endl;
        // pset -> print_permutation_matrix();
        // cout << "local size:  " << locals_sz << endl;
        for (auto local = locals.begin(); local != locals.end(); local ++){
            uint src_svr = (*local) -> get_server_id();
            auto lookup = (*pset).mp.find(src_svr);
            if (lookup == (*pset).mp.end()){
                LOG("error decomposition result");
                exit(1);
            }
            uint dst_svr = lookup -> second;
            (*local) -> restore_one_server2(dst_svr, chnl[pid][lid], ds[pid][lid], d_cpy[pid][lid], (*pset).get_freq());
            lid ++;
        }
        pid++;
    }

    struct scheduling_step_t first_step = {
        .to_server = all2all.p_sets[0].to_server(server_n),
        .from_server = all2all.p_sets[0].from_server(server_n),
        .channel = chnl[0],
    };
    schd_ret.steps.push_back(first_step);

    for (pid = 1; pid != pset_sz; pid ++){
        struct scheduling_step_t cur_step = {
            .to_server = all2all.p_sets[pid].to_server(server_n),
            .from_server = all2all.p_sets[pid].from_server(server_n),
            .channel = chnl[pid],
            .restore = ds[pid - 1],
            .direct_cpy = d_cpy[pid - 1]
        };
        schd_ret.steps.push_back(cur_step);
    }

    struct scheduling_step_t final_step = {
        .restore = ds[pset_sz - 1],
        .direct_cpy = d_cpy[pset_sz - 1]
    };
    schd_ret.steps.push_back(final_step);
    return schd_ret;
}

