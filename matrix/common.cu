//バッファだけ
//エスケープシーケンス用です。
#include <stdarg.h>
#include "common.cuh"
#include "string.h"
#include <iostream>

char _esc_seq_buf[1024];

//
__host__  void pos_printf(int x, int y, char* fmt,...) {
    char buf[256];
    sprintf_s(buf,sizeof(buf), "\x1B[%d;%dH%s", y, x,fmt);
    va_list ap;    va_start(ap, fmt);
    vprintf_s(buf, ap);
//    printf("\x1B[%d;%dH", y, x );
}
__host__ __device__ void esc_clr()
{
    printf("\x1B[2J");
}
__device__ void _pos(int x, int y) {
    printf("\x1B[%d;%dH", y, x);
}

__host__
void dumpGPUInfo()
{

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    std::cout << "Number of SMs: " << numSMs << std::endl;

    {
        int device;
        cudaDeviceProp prop;

        // デバイスの取得
        cudaGetDevice(&device);

        // デバイスプロパティの取得
        cudaGetDeviceProperties(&prop, device);
        printf("CUDA Capability Major/Minor version number: %d.%d\n", prop.major, prop.minor);
    }

}