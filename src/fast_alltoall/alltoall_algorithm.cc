#include "alltoall_algorithm.h"
#include "alltoall_matrix.h"
#include "alltoall_define.h"
#include <iomanip>
#include <pthread.h>
using namespace std;


FastAll2All::FastAll2All(Matrix * _mat, uint _gpu_n){
    mat.copy(_mat);
    uint dim = mat.get_dim();
    hungarian_info.matching.insert(hungarian_info.matching.end(), dim*2, -1);
    hungarian_info.visit.insert(hungarian_info.visit.end(), dim*2, false);
    unordered_set<uint> empty_vector;
    hungarian_info.row_to_col.insert(hungarian_info.row_to_col.end(), dim, empty_vector);
    gpu_n = _gpu_n;
}

void FastAll2All::to_scaled_doubly_stochastic_matrix(){
    mat.get_sdsm_info();
    SDS_mat.copy(&mat);

    if(!SDS_mat.sdsm_info.is_sdsm){
        uint dim = SDS_mat.get_dim();
        uint max_sum = SDS_mat.sdsm_info.max_row_col_sum;
        // original matrix is not SDSM, do the conversion
        for (vector<struct row_col_info_t>::iterator row = SDS_mat.sdsm_info.non_max_row.begin(); row != SDS_mat.sdsm_info.non_max_row.end(); row++){
            for (vector<struct row_col_info_t>::iterator col = SDS_mat.sdsm_info.non_max_col.begin(); col != SDS_mat.sdsm_info.non_max_col.end(); col++){
                if (col -> sum == max_sum)
                    continue;
                uint diff =  max_sum - MAX(row -> sum, col -> sum);
                SDS_mat.add(diff, row -> idx, col -> idx);
                row -> sum += diff;
                col -> sum += diff;
                if (row -> sum == max_sum) {
                    break;
                }          
            }
        }
        SDS_mat.sdsm_info.is_sdsm = true;
        SDS_mat.sdsm_info.non_max_row.clear();
        SDS_mat.sdsm_info.non_max_col.clear();
        // SDS_mat.get_sdsm_info();
    }
}

void FastAll2All::decompose(){
    if (!SDS_mat.valid_sdsm()){
        LOG("error when doing decomposition, must convert matrix to sdsm first!");
        return;
    }
    p_sets.clear(); // store results
    uint freq_sum = 0, max_sum = SDS_mat.sdsm_info.max_row_col_sum;
    while(freq_sum < max_sum){
        update_edges();
        hungarian();
        freq_sum += update_permutation_sets();
    }
}


uint FastAll2All::hungarian(){
    uint match_num = 0;
    uint dim = SDS_mat.get_dim();
    std::fill(hungarian_info.matching.begin(), hungarian_info.matching.end(), -1);
    for (uint u = 0; u < dim; u++){
        if (hungarian_info.matching[u] == -1){
            std::fill(hungarian_info.visit.begin(), hungarian_info.visit.end(), false);
            if(hungarian_dfs(u))
                match_num ++;
        }
    }
    return match_num;
}


bool FastAll2All::hungarian_dfs(uint u){
    for (unordered_set<uint>::iterator col_idx = hungarian_info.row_to_col[u].begin(); col_idx != hungarian_info.row_to_col[u].end(); col_idx ++){
        if (!hungarian_info.visit[*col_idx]){
            hungarian_info.visit[*col_idx] = true;
            if (hungarian_info.matching[*col_idx] == -1 || hungarian_dfs(hungarian_info.matching[*col_idx])){
                hungarian_info.matching[*col_idx] = u;
                hungarian_info.matching[u] = *col_idx;
                return true;
            }
        }
    }
    return false;
}

void FastAll2All::update_edges(){
    uint dim = SDS_mat.get_dim();
    // row vertices id: 0 - dim-1, col vertices id: dim - 2*dim-1
    for (uint i = 0; i < dim; i++){
        for (uint j = 0; j < dim; j++){
            uint col_id = j + dim;
            if (SDS_mat.get(i, j) > 0 ){
                hungarian_info.row_to_col[i].insert(col_id);
            }else if (SDS_mat.get(i, j) == 0){
                hungarian_info.row_to_col[i].erase(col_id);
            }
        }
    }
}

uint FastAll2All::update_permutation_sets(){
    uint dim = SDS_mat.get_dim();
    // row vertices id: 0 - dim-1, col vertices id: 0 - dim-1
    PermutationSet r(1, 1, dim);
    uint min_freq = SDS_mat.get(0, hungarian_info.matching[0] - dim);
    for(uint i = 0; i < dim; i++){
        uint col_id = hungarian_info.matching[i] - dim;
        min_freq = MIN(SDS_mat.get(i, col_id), min_freq);
        r.mp.insert(make_pair(i, col_id));
    }

    for(uint i = 0; i < dim; i++){
        uint col_id = hungarian_info.matching[i] - dim;
        SDS_mat.subtract(min_freq, i, col_id);
    }

    r.set_freq(min_freq);
    p_sets.push_back(r);
    return min_freq;
}

void FastAll2All::print_decomposition(){
    for (vector<PermutationSet>::iterator ps = p_sets.begin(); ps != p_sets.end(); ps++){
        ps -> print_permutation_matrix();
    }
}

bool FastAll2All::verify_decomposition(){
    uint dim = SDS_mat.get_dim();
    Matrix r(dim);
    for (vector<PermutationSet>::iterator ps = p_sets.begin(); ps != p_sets.end(); ps++){
        for (uint i = 0; i < dim; i++){
            uint non_empty_col_id = ps -> mp[i];
            r.add(ps->get_freq(), i, non_empty_col_id);
        }
    }
    r.get_sdsm_info();
    to_scaled_doubly_stochastic_matrix();
    return SDS_mat.equal_to(&r);
}

void PermutationSet::print_permutation_matrix(){
    cout << "Permutation Matrix, dim: " << dim << endl;
    for(uint i = 0; i < dim; i ++){
        uint non_empty_col_id = mp[i];
        for (uint j = 0; j < dim; j++){
            cout << setw(10);
            if (non_empty_col_id == j){
                cout << frequency * scaling_factor;
            }else{
                cout << "0";
            }
        }
        cout << endl;
     }
}


vector<uint> PermutationSet::to_server(uint server_n){
    vector<uint> r;
    for (uint i = 0; i < server_n; i++){
        auto lookup = mp.find(i);
        if (lookup == mp.end()){
            LOG("error decomposition result");
            exit(1);
        }
        r.push_back(lookup->second);
    }
    return r;
}
vector<uint> PermutationSet::from_server(uint server_n){
    vector<uint> r;
    for (uint i = 0; i < server_n; i++){
        r.push_back(0);
    }
    for (uint i = 0; i < server_n; i++){
        auto lookup = mp.find(i);
        if (lookup == mp.end()){
            LOG("error decomposition result");
            exit(1);
        }
        r[lookup->second] = i; 
    }
    return r;
}


