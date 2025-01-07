#include <cuda_runtime.h>
#include "layer.cuh"
#include <algorithm>


//ノードを割り付けます。
__host__
void layer::alloc_nodes(int layer_num , int n_nodes ,int act_type/*=ACT_NOP*/, layer*p_prev_layer /*=0*/ )
{
    nodes.alloc(n_nodes);
    //一番前はｗを設定しない。
    if (p_prev_layer) {
        for (int i = 0; i < nodes.size; ++i) {
            nodes.cpu[i].alloc_w(p_prev_layer->nodes.size);
            nodes.cpu[i].set_act_type(act_type);
        }
    }
}


//GPUへノードを転送する。
__host__
void layer::Transfer_contained_members_to_GPU()
{
    for (int i = 0; i < nodes.size; ++i) {      //nodeのメンバをメモリ転送する指示です。
        nodes.cpu[i].Transfer_contained_members_to_GPU();
    }
    nodes.Transfer_to_GPU();                    //node配列自身のメモリ転送です。
}

//CPUへノード転送する。
__host__
void layer::Transfer_contained_members_to_CPU()
{
    for (int i = 0; i < nodes.size; ++i) {
        nodes.cpu[i].Transfer_contained_members_to_CPU();         
    }
    nodes.Transfer_to_CPU();                    //node自身のメモリ転送。
}
//ダンプします。
__host__ __device__
void layer::dump(int l,int locate)const
{
    for (int i=0; i < nodes.size; ++i) {
        nodes(locate,i).dump(l,i);
    }
}
//-----------------------------------------------------------------------------------------------------
//  max
//-----------------------------------------------------------------------------------------------------
//リダクションは使えないとして、threadIdx.xはノードの数分ぴったりで入ってくる。
//ここでついでにsumも計算してしまう。
#define _max(a,b)   ((a>b)?a:b)
__device__
void layer::gpu_calc_softmax_max_sum()
{
    __shared__ float max_arr[256];  //計算用
    _Assert(nodes.size < (sizeof(max_arr) / sizeof(float)), "layer::calc_max() : n nodes over");

    //thread.x 0-15までしか
    if (threadIdx.y == 0 && (threadIdx.x < nodes.size)) {
        max_arr[threadIdx.x] = nodes(GPU, threadIdx.x).a;

    }
    __syncthreads();

    // 段階的な比較で最大値を求める
    for (int stride = 1; stride < nodes.size; stride *= 2) {
        if (threadIdx.y == 0) {
            if ((threadIdx.x % (2 * stride) == 0) && (threadIdx.x + stride < nodes.size)) {        //ストライド単位で評価する＆スレッドがストライドの先もある場合のみ。
                max_arr[threadIdx.x] = _max(max_arr[threadIdx.x], max_arr[threadIdx.x + stride]);
            }
        }
        __syncthreads();  // 各ステップで全スレッドを同期
    }
    //ここまでで、max_arr[0]に、threadIdx.x : 0 - 一番大きい偶数までの評価が終わっている。ノードの数が奇数の場合、一番最後の値が評価されていないので、評価する。
    if(threadIdx.y==0 && threadIdx.x==0){
        if(nodes.size%2!=0){
            max_arr[0] =  _max( max_arr[0] , max_arr[nodes.size-1]);
        }
#if 1     //検算します。
        {
            float m = nodes(GPU, 0).a;
            for (int i = 1; i < nodes.size; ++i) {
                float a = nodes(GPU, i).a;
                if (a > m) {
                    m =a;
                }
            }
            _Assert(m == max_arr[0], "calc err");
        }
#endif
    }
    __syncthreads();
    //--------------------------------------------------------------------
    //  ここから先はΣ(exp(a-max))の計算です。
    //--------------------------------------------------------------------
    {
        __shared__ float _sum;
        __shared__ float _softmax[256];
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            _sum = 0.0f;
        }
        __syncthreads();
        if (threadIdx.y == 0 && (threadIdx.x < nodes.size)) {
            atomicAdd(&_sum ,expf(nodes(GPU, threadIdx.x).a - max_arr[0]));
#if 1
            _softmax[threadIdx.x] = expf(nodes(GPU, threadIdx.x).a - max_arr[0]);
#endif
        }
        __syncthreads();
#if 1   //検算する。
        if(threadIdx.y==0 && threadIdx.x==0){
            float s=0.0;
            if(threadIdx.y == 0 && threadIdx.x==0){
                for(int i = 0; i < nodes.size;  ++i ){
                    s+= expf(nodes(GPU, i).a - max_arr[0]);
                }
                _Assert(abs(_sum-s)<0.0001,"calc sum error");
            }
        }
#endif
        __syncthreads();
        sum = _sum;             //sumはここです。
    }
    max = max_arr[0];     //maxはここです。
    __syncthreads();
}

//-----------------------------------------------------------------------------------------------------
//  activate処理です。
//-----------------------------------------------------------------------------------------------------
__device__ 
void layer::gpu_activate()
{
    if (nodes(GPU, 0).act_type == ACT_SOFTMAX) {
        gpu_calc_softmax_max_sum();
    }
    __syncthreads();
    if (threadIdx.y ==0 && threadIdx.x < nodes.size) {
            nodes(GPU,threadIdx.x).activate(this);
    }
    __syncthreads();
}