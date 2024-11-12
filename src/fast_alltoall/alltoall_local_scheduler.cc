#include "alltoall_local_scheduler.h"
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <hip/hipruntime.h>


using namespace std;

LocalScheduler::LocalScheduler(uint* _data, uint _gpu_n, uint _server_n, uint _server_id){
    gpu_n = _gpu_n;
    server_n = _server_n;
    server_id = _server_id;
    uint dim = gpu_n * server_n;
    hipMallocManaged((void**) &data, sizeof(uint*) * gpu_n);
    hipMallocManaged((void**) &balanced_data, sizeof(uint*) * gpu_n);
    hipMallocManaged((void**) &data_after_balance, sizeof(data_t*) * gpu_n);
    // data = new uint*[gpu_n];
    // balanced_data = new uint*[gpu_n];
    // data_after_balance = new data_t*[gpu_n];
    uint idx = 0;
    for (uint i = 0; i < gpu_n; i++){
        hipMallocManaged(data[i], sizeof(uint) * dim);
        hipMallocManaged(balanced_data[i], sizeof(uint) * dim);
        hipMallocManaged(data_after_balance[i], sizeof(data_t) * dim);
        // data[i] = new uint[dim];
        // balanced_data[i] = new uint[dim];
        // data_after_balance[i] = new data_t[dim];
        for (uint j = 0; j < dim; j++){
            data[i][j] = _data[idx];
            balanced_data[i][j] = _data[idx];
            for (uint z = 0; z < MAX_GPU_PER_SERVER; z++){
                data_after_balance[i][j].sz[z] = 0;
                data_after_balance[i][j].offset[z] = 0;
            }
            data_after_balance[i][j].sz[i] = _data[idx];
            data_after_balance[i][j].sum = _data[idx];
            idx++;
        }
    }
    hipMallocManaged((void**)&server2server_data, sizeof(uint) * server_n);
    hipMemset(server2server_data, 0, server_n * sizeof(uint));
    hipMallocManaged((void**)&row_sum, sizeof(uint) * gpu_n * server_n);
    hipMemset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    hipMallocManaged((void**)&intrinsic_all2all, sizeof(uint) * gpu_n * gpu_n);
    hipMemset(intrinsic_all2all, 0, gpu_n * gpu_n * sizeof(uint));
    // server2server_data = new uint[server_n];
    // memset(server2server_data, 0, server_n * sizeof(uint));
    // row_sum = new uint[gpu_n * server_n];
    // memset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    // intrinsic_all2all = new uint[gpu_n * gpu_n];
    // memset(intrinsic_all2all, 0, gpu_n * gpu_n * sizeof(uint));
    prepare_load_balance();
}

LocalScheduler::~LocalScheduler(){
    for (uint i = 0; i < gpu_n; i++){
        hipFree(data[i]);
        hipFree(balanced_data[i]);
        hipFree(data_after_balance[i]);
        // delete[] data[i];
        // delete[] balanced_data[i];
        // delete[] data_after_balance[i];
    }
    hipFree(data);
    hipFree(balanced_data);
    hipFree(data_after_balance);
    hipFree(server2server_data);
    hipFree(row_sum);
    hipFree(intrinsic_all2all);
    // delete[] data;
    // delete[] balanced_data;
    // delete[] data_after_balance;
    // delete[] server2server_data;
    // delete[] row_sum;
    // delete[] intrinsic_all2all;
}

void LocalScheduler::prepare_load_balance(){
    hipMemset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    // memset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    for (uint i = 0; i < server_n; i++){
        if (i == server_id){
            server2server_data[i] = 0;
            for (uint j = 0; j < gpu_n; j++){
                for (uint k = 0; k < gpu_n; k++){
                    intrinsic_all2all[j * gpu_n + k] = data[j][server_id * gpu_n + k];
                }
            }
            continue;
        }
        for (uint j = 0; j < gpu_n; j++){
            // for each row at each tile
            row_sum[i * gpu_n + j] = 0;
            for (uint k = 0; k < gpu_n; k++){
                row_sum[i * gpu_n + j] += data[j][i * gpu_n + k];
            }
        }

        uint row_avg = 0;
        for (uint k = 0; k < gpu_n; k++){
            row_avg += row_sum[i * gpu_n + k];
        }
        server2server_data[i] = (row_avg + gpu_n - 1) / gpu_n;
    }
    // print();
    // for (uint i = 0; i < server_n; i++){
    //     cout << "to server " << i << ":" << server2server_data[i] << endl;
    // }
}

void LocalScheduler::balance_one_server2(uint to_server_id, BalancePtr r){
    vector<uint> smaller_row;
    vector<uint> bigger_row;
    if (to_server_id == server_id){
        return;
    }

    for (uint i = 0; i < gpu_n; i++){
        if (row_sum[to_server_id * gpu_n + i] < server2server_data[to_server_id]){
            smaller_row.push_back(i);
        }else if (row_sum[to_server_id * gpu_n + i] > server2server_data[to_server_id]){
            bigger_row.push_back(i);
        }
    }

    for (auto big_row = bigger_row.begin(); big_row != bigger_row.end(); big_row++){

        int rm_data = row_sum[to_server_id * gpu_n + *big_row] - server2server_data[to_server_id];
        for (auto small_row = smaller_row.begin(); small_row != smaller_row.end();){
            for (uint j = 0; j < gpu_n; j++){
                // check each element of the big row
                int mv_data = MIN(MIN(rm_data, data_after_balance[*big_row][to_server_id * gpu_n + j].sum), server2server_data[to_server_id] - row_sum[to_server_id * gpu_n + *small_row]);
                // cout << "LB scheduler, mv data: " << mv_data <<", channel: "<< j << ", src gpu: " << *big_row << ", dst gpu: " << *small_row << endl;

                if (mv_data == 0){
                    continue;
                }
                rm_data -= mv_data;
                // big row col j ====> small row col j via balance big row -> small row
                row_sum[to_server_id * gpu_n + *small_row] += mv_data;
                row_sum[to_server_id * gpu_n + *big_row] -= mv_data;
                r[(*big_row) * gpu_n + (*small_row)].sz[j] += mv_data;

                // cout << "lb server" << server_id << ", dst server" << to_server_id << ", mv data: " << mv_data<<", big row: " << *big_row << ", small row: " << *small_row <<", lb dst gpu: " << j << endl;
                data_after_balance[*big_row][to_server_id * gpu_n + j].sz[*big_row] -= mv_data;
                data_after_balance[*big_row][to_server_id * gpu_n + j].sum -= mv_data;
                data_after_balance[*small_row][to_server_id * gpu_n + j].offset[*big_row] = data_after_balance[*big_row][to_server_id * gpu_n + j].offset[*big_row];
                data_after_balance[*big_row][to_server_id * gpu_n + j].offset[*big_row] += mv_data;
                data_after_balance[*small_row][to_server_id * gpu_n + j].sz[*big_row] += mv_data;
                data_after_balance[*small_row][to_server_id * gpu_n + j].sum += mv_data;
                if (rm_data == 0){
                    break;
                }
            }
            if (rm_data == 0){
                break;
            }

            if (row_sum[to_server_id * gpu_n + *small_row] == server2server_data[to_server_id]){
                small_row = smaller_row.erase(small_row);
            }else{
                small_row ++;
            }
        }

        if (smaller_row.empty()){
            break;
        }
    }
}

// void LocalScheduler::balance_one_server(uint to_server_id, BalancePtr r){

//     vector<uint> smaller_row;
//     vector<uint> bigger_row;

//     for (uint i = 0; i < gpu_n; i++){
//         if (row_sum[to_server_id * gpu_n + i] < server2server_data[to_server_id]){
//             smaller_row.push_back(i);
//         }else if (row_sum[to_server_id * gpu_n + i] > server2server_data[to_server_id]){
//             bigger_row.push_back(i);
//         }
//     }

//     for (auto big_row = bigger_row.begin(); big_row != bigger_row.end(); big_row++){

//         uint rm_data = row_sum[to_server_id * gpu_n + *big_row] - server2server_data[to_server_id];
//         for (auto small_row = smaller_row.begin(); small_row != smaller_row.end();){
//             for (uint j = 0; j < gpu_n; j++){
//                 // check each element of the big row
//                 uint mv_data = MIN(MAX_BUFFER_SIZE_PER_RANK - balanced_data[*small_row][to_server_id * gpu_n + j], MIN(MIN(rm_data, balanced_data[*big_row][to_server_id * gpu_n + j]), server2server_data[to_server_id] - row_sum[to_server_id * gpu_n + *small_row]));
//                 rm_data -= mv_data;
//                 // big row col j ====> small row col j via balance big row -> small row
//                 row_sum[to_server_id * gpu_n + *small_row] += mv_data;
//                 row_sum[to_server_id * gpu_n + *big_row] -= mv_data;
//                 balanced_data[*big_row][to_server_id * gpu_n + j] -= mv_data;
//                 balanced_data[*small_row][to_server_id * gpu_n + j] += mv_data;
//                 r[(*big_row) * gpu_n + (*small_row)] += mv_data;
//             }
//             if (rm_data == 0){
//                 break;
//             }

//             if (row_sum[to_server_id * gpu_n + *small_row] == server2server_data[to_server_id]){
//                 small_row = smaller_row.erase(small_row);
//             }else{
//                 small_row ++;
//             }
//         }

//         if (smaller_row.empty()){
//             break;
//         }
//     }
// }

void LocalScheduler::restore_one_server2(uint to_server_id, vector<ChannelPtr> channel, vector<RestorePtr> r, vector<DirectCpyPtr> dcpy, uint freq){

    if (to_server_id == server_id){
        return;
    }
    // uint * row_transfer = new uint[gpu_n];
    uint * row_transfer;
    hipMallocManaged((void**) &row_transfer, sizeof(uint) * gpu_n);
    for (uint i = 0; i < gpu_n; i++){    // src gpu

        row_transfer[i] = 0;
        for (uint j = 0; j < gpu_n; j++){   // dst gpu
            for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu++){
                int transfer = MIN(freq - row_transfer[i], data_after_balance[i][to_server_id * gpu_n + j].sz[from_gpu]);
                if (data_after_balance[i][to_server_id * gpu_n + j].sz[from_gpu] == 0){
                    continue;
                }
                row_transfer[i] += transfer;
                channel[i][ j * gpu_n + from_gpu] += transfer;
                data_after_balance[i][to_server_id * gpu_n + j].sz[from_gpu] -= transfer;
                row_sum[to_server_id * gpu_n + i] -= transfer;
                if (i != j){    // Transfer is Server m's GPUi --> Server n's GPUi, need to dispatch data from m's GPUi --> n's GPUj if i not equal j
                    //     cout << "restore server" << server_id << ", dst server" << to_server_id << ", restore data: " << transfer<<", src gpu: " << i << ", dst gpu: " << j << endl;
                    // cout << "src : "<< i << ", dst: " << j << ", from gpu: " << from_gpu <<", size: " << transfer << endl;
                    r[i][j * gpu_n + from_gpu].sz += transfer;
                    r[i][j * gpu_n + from_gpu].offset = data_after_balance[i][to_server_id * gpu_n + j].offset[from_gpu];
                }else{
                    dcpy[i][from_gpu].sz += transfer;
                    dcpy[i][from_gpu].offset = data_after_balance[i][to_server_id * gpu_n + j].offset[from_gpu];
                }
                data_after_balance[i][to_server_id * gpu_n + j].offset[from_gpu] += transfer;
                if (row_transfer[i] == freq){
                    break;
                }
            }
            if (row_transfer[i] == freq){
                break;
            }
        }

    }
    hipFree(row_transfer);
    // delete[] row_transfer;

}


// void LocalScheduler::restore_one_server(uint to_server_id, RestorePtr r, DirectCpyPtr dcpy, uint freq){
//     uint * row_transfer = new uint[gpu_n];

//     for (uint i = 0; i < gpu_n; i++){

//         row_transfer[i] = 0;
//         for (uint j = 0; j < gpu_n; j++){
//             uint transfer = MIN(freq - row_transfer[i], balanced_data[i][to_server_id * gpu_n + j]);
//             row_transfer[i] += transfer;
//             balanced_data[i][to_server_id * gpu_n + j] -= transfer;
//             row_sum[to_server_id * gpu_n + i] -= transfer;
//             if (i != j){    // Transfer is Server m's GPUi --> Server n's GPUi, need to dispatch data from m's GPUi --> n's GPUj if i not equal j
//                 r[i * gpu_n + j] += transfer;
//             }else{
//                 dcpy[i] = transfer;
//             }
//             if (row_transfer[i] == freq){
//                 break;
//             }
//         }

//     }
//     delete[] row_transfer;
// }

void LocalScheduler::print(uint dst_server_id){
    cout << "server "<< server_id << " to server " << dst_server_id << endl;
    for (uint i = 0; i < gpu_n; i++){
        for (uint j = 0; j < gpu_n; j++){
            cout << setw(10);
            cout << data[i][dst_server_id * gpu_n + j];
        }
        cout << endl;
    }
}


void LocalScheduler::print(){
    uint dim = gpu_n * server_n;

    cout << "original matrix: " << endl;
    for (uint i = 0; i < gpu_n; i++){
        for (uint j = 0; j < dim; j++){
            cout << setw(10);
            cout << data[i][j];
        }
        cout << endl;
    }

    // cout << endl << "balanced matrix: " << endl;
    // for (uint i = 0; i < gpu_n; i++){
    //     for (uint j = 0; j < dim; j++){
    //         cout << setw(10);
    //         cout << balanced_data[i][j];
    //     }
    //     cout << endl;
    // }

    // cout << endl << "----------------" << endl << "Intra All2All" << endl;
    // for (auto it = intra_all2all.begin(); it != intra_all2all.end(); it++){
    //     cout << "server " << server_id << " to server " << it->first << endl;
    //     cout << "balance:" << endl;
    //     for (uint i = 0; i < gpu_n; i++){
    //         for (uint j = 0; j < gpu_n; j++){
    //             cout << setw(10);
    //             cout << it->second.balance[i*gpu_n + j];
    //         }
    //         cout << endl;
    //     }
    //     cout << endl<< "dispatch:" << endl;
    //     for (uint i = 0; i < gpu_n; i++){
    //         for (uint j = 0; j < gpu_n; j++){
    //             cout << setw(10);
    //             cout << it->second.dispatch[i*gpu_n + j];
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }
}

void print_matrix(uint * data, uint m, uint n){ //width m, height n
    cout << "--------------------------------------"<<endl;
    for(uint i = 0; i < n; i ++){
        for (uint j = 0; j < m; j++){
            cout << setw(10);
            cout << data[i * m + j];
        }
        cout << endl;
    }
    cout << "--------------------------------------"<<endl;
}