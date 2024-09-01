
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "common.h"

//#include "node.h"
#include "net.h"

void dumpGPUInfo();

#define     NET_N_LAYER     3

//入力データ、回答データもGPUへ送り、GPU内でループさせたい。
//学習が終わるまで

__global__ void fwd_kernel(net<NET_N_LAYER>* _net)
{
    int cur_idx    = threadIdx.y;          //値を求めるノードの番号です。ブロックに割り付けます。
    int prev_idx   = threadIdx.x;         //全結合を行う全段のノードです。
    //このスレッドが、次レイヤーのどのインデックスのノードに対し担当するか：block_idx
    for (int l = 1; l < _net->n_layers; ++l) {
        layer& L_pre = _net->layers[l - 1], L_cur = _net->layers[l];
        node& prev = L_pre.nodes.gpu[prev_idx];
        node& cur = L_cur.nodes.gpu[cur_idx];

        if ( cur_idx < L_cur.nodes.size && prev_idx < L_pre.nodes.size) {


            // 前段ノードすべてから、今回のノード cur_node_idxのノードへ全結合が行われる。
            atomicAdd(&cur.a, prev.y * cur.w.gpu[prev_idx]);
        }

        __syncthreads();  // 全スレッドが同期する場所は、ループ内の条件外に配置する

        //=============================================================================
        //  activate
        //=============================================================================
        if (prev_idx == 0 && cur_idx < L_cur.nodes.size) {      //  全結合が終わったので
            node& cur = L_cur.nodes.gpu[cur_idx];
            cur.a += cur.b;
            cur.y = cur.activate();
        }
        __syncthreads();  // さらに必要なら、もう一度同期を挟む
    }
    //ここで、
    //最終段のdE/daを求めます。
    _net->loss_softmax_with_crossentropy();     //
    //
}

int main() {

    dumpGPUInfo();

    net<NET_N_LAYER> myNet(3,5,2);  // 3ノード、5ノード、2ノードを持つネットワークを作成
    //net自身をgpuへコピーします。
    //input作る
    for (int i = 0; i < myNet.layers[0].nodes.size; ++i) {
        myNet.layers[0].nodes.cpu[i].y = .5 * (1+i);
    }

    net<NET_N_LAYER>* pnet = myNet.Transfer_to_GPU();        //netの中身を転送します。

    dim3 threads(16, 16, 1);                             //スレッドとしてパーセプトロンを動作させるようにします。これがバッチ数分ある。
    dim3 blocks(1, 1, 1);                              //とりあえずバッチ分をこちらにこれが128個あるはず
    fwd_kernel << < blocks, threads >> > (pnet);

    cudaDeviceSynchronize();                            //


    //GPU→CPUへ計算結果を転送する。
    myNet.Transfer_to_CPU();                            //
    myNet.dump();


//    if ((err = cudaGetLastError())!= cudaSuccess)    {      printf("Error: %s\n", cudaGetErrorString(err));         return;    }
    return 0;
}

void dumpGPUInfo()
{

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    std::cout << "Number of SMs: " << numSMs << std::endl;

}

