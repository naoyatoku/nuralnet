#pragma once
#include "common.cuh"
#include "node.cuh"

enum{
    REQ_CALC_MAX,
};

class layer
{
public:
    cpu_gpu_mem<node> nodes;  //
    __host__ layer() { ; }

    //ノード割付け
    __host__ void alloc_nodes(int layer_num , int n_nodes ,int act_type=ACT_NOP, layer*p_prev_layer =0 ) ;

    __host__ void Transfer_contained_members_to_GPU();      //ノードをGPUへ転送
    __host__ void Transfer_contained_members_to_CPU();      //
    __host__ __device__ void dump(int l,int locate=CPU)const ;

    //各ノードからのリクエストを受け付ける。
    __device__  float request(int cmd, ...)const;
};
