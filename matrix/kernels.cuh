#pragma once
#include "cuda_runtime.h"

class net;

//入力データ、回答データもGPUへ送り、GPU内でループさせたい。
//---------------------------------------------------
//  kernel.cu
//---------------------------------------------------
__global__ void fwd_kernel(net* _net);


//---------------------------------------------------
//  sub_kernel.cu
//---------------------------------------------------
__device__ float calc_max( const node *nd , int size);
