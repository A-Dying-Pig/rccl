#pragma once

#include <iostream>
#include <vector>
#include <unordered_set>
#include <map>
#include "alltoall_matrix.h"
using namespace std;


struct hungarian_info_t{
    vector<int> matching;
    vector<bool> visit;
    vector<unordered_set<uint> > row_to_col; // the col vertices that each row vertice connects to
};

struct pset_t{
    vector<uint> mp;
    uint freq;
};


class PermutationSet{
private:
    uint frequency;
    uint scaling_factor;
    uint dim;
public:
    map<uint, uint> mp;     // mapping between row vertice and col vertice, both indexed from 0-dim-1
    PermutationSet(uint _freq = 1, uint _sf = 1, uint _dim = 0):frequency(_freq), scaling_factor(_sf), dim(_dim){}
    ~PermutationSet(){}
    void set_freq(uint freq){frequency = freq;}
    uint get_freq(){return frequency * scaling_factor;}
    void print_permutation_matrix();
    vector<uint> to_server(uint server_n);
    vector<uint> from_server(uint server_n);
};


class FastAll2All{
private:
    Matrix mat;     // original input matrix
    Matrix SDS_mat; // scaled doubly stochastic matrix
    struct hungarian_info_t hungarian_info;  //metadata used by hungarian algorithm
    uint gpu_n;

public:
    vector<PermutationSet> p_sets; // permutation sets, storing decomposition results
    FastAll2All(Matrix * _mat, uint _gpu_n);
    ~FastAll2All(){}
    void print(){mat.print();SDS_mat.print();}
    void to_scaled_doubly_stochastic_matrix();
    void update_edges();
    uint update_permutation_sets();
    void decompose();
    uint hungarian();
    bool hungarian_dfs(uint u);
    bool verify_decomposition();
    void print_decomposition();
};
