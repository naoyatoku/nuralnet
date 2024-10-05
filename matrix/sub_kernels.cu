
#include "cuda_runtime.h"
#include "common.cuh"
#include "node.cuh"




//-----------------------------------------------------------------------------------------------------------------
// netを使用するnodeメンバ関数をここに
//-----------------------------------------------------------------------------------------------------------------
//これカーネル。
__global__ void _calc_max_kernel(const node* nd, int n_node,float *_max)
{
    int x = threadIdx.x;                //本スレッドが担当するノード
    //一番近い2のs累乗の数字を探す。
    __shared__ float max[256];          //※このメモリはカーネルが終わるまでなくなりません。また、固定のサイズでしか取れません。
    __shared__ int _n;                  //最大値を求める際のリダクションの数（2の累乗の数：でないとリダクションが成立しない）
    //リダクションの数を決めます.n_node に一番近い
    if (x == 0) {
        _n = 1;
        while (_n < n_node) {   //2倍していき、n_ndに一番近いを探します。
           _n <<= 1;
        }
        _Assert(sizeof(max) / sizeof(float) > _n, "max size is not enough");
    }
    __syncthreads();
    //ここで_nが定まりました。
    // max初期化します。
    if (x < _n) {
        max[x] = (x < n_node) ? nd[x].a : -10000000000.0;
    }
    __syncthreads();
    //リダクションしていきます。
    for (int n = _n / 2; n > 0; n /= 2) {                                  //N=7 : n=3         n=1
        if (x < n) {                                                        //thread.xは0,1,2 
            if (max[x] < max[x + n]) {
                max[x] = max[x + n];
            }
        }
        __syncthreads();
    }
    //ここでmax[0]に最大値はいっているはず
#if 1  //debug（検算です）
    {
        if (x == 0) {
            float m = -100.0;
            for (int i = 0; i < n_node; ++i) {
                if (nd[i].a > m) {
                    m = nd[i].a;
                }
            }
            _Assert(max[0] == m, "max different");
        }

    }
#endif
    __syncthreads();

    *_max  = max[0];     //これが小舘です
    return;

}
//これがカーネルを呼び出します。呼び出したデバイスの状態に関係なく必要なスレッドを呼びたいため。
__device__ float calc_max( const node *nd , int size)
{
    _Assert(size < 512, "calcmax() thread num overflw");    //一番近い2のべき乗のスレッド数が必要なためスレッドｘの最大値は1024なので512を最大とします。これより大きなものの比較はまた別に考えます。
    float max;
    dim3 threads(size*2, 1, 1);                          //もしバッチでやりたい場合はブロック数またはyを増やすか
    dim3 blocks(1, 1, 1);
    _calc_max_kernel << < blocks, threads >> > (nd,size,&max);   //
    return max;
}
