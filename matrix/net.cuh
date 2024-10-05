#pragma once

#include "common.cuh"
#include "layer.cuh"

//損失関数タイプ
enum {
	LOSS_MEAN_SQUARE,
	LOSS_ENTROPY,
	LOSS_HUBER,
    LOSS_SOFTMAX_WITH_CROSSENTROPY , //
    LOSS_NOP,                      //なにもしない
};

//テンプレートを使うことで、関数の定義が難しくなるので、いったんやめてみる。
class net
{
public:
    cpu_gpu_mem<layer>ly;  //レイヤーです。
    //constructor
    __host__ net() = default;                    // デフォルトコンストラクタ
    __host__ void construct(int n_layers, int n_input, int n_mid_nodes, int act_type, int n_output_nodes, int output_act_type, int loss_type);

    //memory trans
    __host__ void Transfer_contained_members_to_GPU();
    __host__ void Transfer_contained_members_to_CPU();

    //dump
    __host__ __device__ void dump(int locate=CPU)const;

    //入力と、正解の組み合わせをセットして、学習を行えるようにする。
    __host__ void set_input_answer(float *in,int in_size , float *answer,int ans_size);
//    __device__ void loss_softmax_with_crossentropy();
};

