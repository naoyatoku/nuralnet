#pragma once
#include "common.cuh"
#include "node.cuh"

enum{
    REQ_CALC_MAX,
};

class layer
{
public:
    cpu_gpu_mem<node> nodes;    //
    //ソフトマックス用の計算値
    float max;                  //
    float sum;                  //

    //layerのコンストラクタ時点では
    __host__ layer() { 
        max = sum = 0.0;

    }

    //ノード割付け
    __host__ void alloc_nodes(int layer_num , int n_nodes ,int act_type=ACT_NOP, layer*p_prev_layer =0 ) ;

    __host__ void Transfer_contained_members_to_GPU();      //ノードをGPUへ転送
    __host__ void Transfer_contained_members_to_CPU();      //
    __host__ __device__ void dump(int l,int locate=CPU)const ;

    //gpu専用命令
    __device__ void    gpu_calc_softmax_max_sum();     //softmax用のmaxとsumを実行する。
    __device__ void     gpu_activate();
};
