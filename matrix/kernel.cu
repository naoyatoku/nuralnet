
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "common.cuh"
#include "net.cuh"

#define     NET_N_LAYER     3

//入力データ、回答データもGPUへ送り、GPU内でループさせたい。
//学習が終わるまで

__global__ void fwd_kernel(net* _net)
{
    int cur_idx    = threadIdx.x;          //値を求めるノードの番号です。ブロックに割り付けます。
    int prev_idx   = threadIdx.y;         //全結合を行う前段のノードです。
    //このスレッドが、次レイヤーのどのインデックスのノードに対し担当するか：block_idx
    for (int l = 1; l < _net->ly.size; ++l) {
        layer& L_pre = _net->ly(GPU,l - 1), L_cur = _net->ly(GPU,l);
        node& prev = L_pre.nodes.gpu[prev_idx];
        node& cur = L_cur.nodes.gpu[cur_idx];
        //============================================================================
        //   affine
        //============================================================================
        if ( cur_idx < L_cur.nodes.size && prev_idx < L_pre.nodes.size) {
            // 前段ノードすべてから、今回のノード cur_node_idxのノードへ全結合が行われる。
            atomicAdd(&cur.a, prev.y * cur.w.gpu[prev_idx]);
            if(prev_idx==0){        //対象となるノードんに対して一つのスレッドだけが行います。
                cur.a += cur.b;     //bias
            }
        }

        __syncthreads();  // 全スレッドが同期する場所は、ループ内の条件外に配置する

        //=============================================================================
        //  activate
        //=============================================================================
        //ここはthreadIdx.xのみ、かつ同レイヤーのノード数しか入ってこない。そのため、内部でmaxの計算を擦る際に、ノード数以上のスレッドが必要になってしまう
        if (prev_idx == 0 && cur_idx < L_cur.nodes.size) {      //  全結合が終わったので  : prev_idx==0 はthread.x , 各ノードは threadIdx.y
            //これからの処理は対象レイヤーの各ノード分のスレッドのみ
            node& cur = L_cur.nodes.gpu[cur_idx];
            //
#if 1
            if(l==2)_net->dump(GPU);
#endif
            cur.y = cur.activate(&L_cur);     //ノードのactivateでsoftmaxの場合、全部のmaxとsumが必要
        }
        __syncthreads();  // さらに必要なら、もう一度同期を挟む
    }
#if 1
    if (threadIdx.y < _net->ly.size ) {
        if (threadIdx.x < _net->ly(GPU,threadIdx.y).nodes.size) {
            int l = threadIdx.y; int n = threadIdx.x;
            printf("[%d][%d]a;%fy;%f\n", l , n , _net->ly(GPU,l).nodes(GPU,n).a, _net->ly(GPU,l).nodes(GPU,n).y );
        }
    }
    __syncthreads();  // さらに必要なら、もう一度同期を挟む

#endif
    //ここで、
    //最終段のdE/daを求めます。
//    _net->loss_softmax_with_crossentropy();       //
//    _net->loss_softmax_with_crossentropy();
//    _net->test();
//    _net->loss_softmax_with_crossentropy();
}

