
#include "cuda_runtime.h"
#include <iostream>

#include "net.cuh"
#include "common.cuh"
#include "kernels.cuh"

int main() {

    dumpGPUInfo();

    //netをセットアップする。面倒か。。。？
    cpu_gpu_mem<net> _net(1);
    _net().construct(3, 3, 5, ACT_RELU, 7, ACT_SOFTMAX, LOSS_SOFTMAX_WITH_CROSSENTROPY);
    //入力を作らないと、とりあえず適当な数値です。
    for (int i = 0; i < _net().ly(CPU, 0).nodes.size; ++i) {
        _net().ly(CPU, 0).nodes(CPU, i).y = (i * 1) * 0.5;
    }



    _net.cpu->dump();
    //GPUへの転送作業です。
    _net().Transfer_contained_members_to_GPU();
    _net.Transfer_to_GPU();


    dim3 threads(16, 16, 1);                             //スレッドとしてパーセプトロンを動作させるようにします。これがバッチ数分ある。
    dim3 blocks(1, 1, 1);                              //とりあえずバッチ分をこちらにこれが128個あるはず
    fwd_kernel << < blocks, threads >> > (_net.gpu);

    cudaDeviceSynchronize();                            //


    //GPU→CPUへ計算結果を転送する。
//    myNet.Transfer_to_CPU();                            //
//    myNet.dump();


//    if ((err = cudaGetLastError())!= cudaSuccess)    {      printf("Error: %s\n", cudaGetErrorString(err));         return;    }
    return 0;
}


