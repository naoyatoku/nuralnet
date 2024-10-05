
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.cuh"


//class net;

enum{
    ACT_NOP,        //inputレイヤー用、なにも行わない
    ACT_RELU,
    ACT_SOFTMAX,
};


class layer;        //親レイヤーのポインタを持っておきます。

struct node
{
    float a;                // 入力
    float dE_da;            //
    float y;                //    
    cpu_gpu_mem<float>w;    //全段nodeの重みづけです
    float b;               //バイアス項です。
    //acivate type
    int     act_type;

    __device__ float activate_softmax(const layer* parent);
    __device__ float d_activate_softmax();

    __device__ float activate(const layer *parent);                // 活性化
    __device__ float d_activate();
    __host__ node() : a(0.0), dE_da(0.0), y(0.0),b(0.0),act_type(ACT_RELU) {     a=0.0;    }

    __host__ void set_act_type(int _act)                { act_type = _act;      }

    //重みづけ
    //matrixを持つようにする。
    //このノードに対する前段ノード数分の重み（行列の１行分）現段ノード数分の行が行がある。行列となる。
    //
    //  
    //バックプロパゲーションのときにwのupdate1もこのほうがわかりやすいんじゃないか。
    __host__ void alloc_w(int n_prev_nodes);    //重みに適当な値をセットする。
    __host__ void __w_init_Xavier(void);
    __host__ void __w_init_He(void);
    __host__ void __w_init_std(void);

    //これは、自分に所属するcpu_gpuメンバをメンバcpu<->gpuメモリ転送
    __host__ void Transfer_contained_members_to_GPU();
    __host__ void Transfer_contained_members_to_CPU();
    //dump
    __host__ __device__ void dump(int l,int n)const;
};


