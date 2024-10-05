#ifndef __COMMONFUNC_H__
#define __COMMONFUNC_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <new>  //
using namespace std;



//net.h と activates.h 両方で定義を参照する必要があったので、ここに定義しています。
#define     _MAX_UNITS      1024        //128   units each layer max
//(windows)
#define		M_PI			(3.1415)


__host__ __device__ static void _Assert( bool a, const char*comment ,...){
    //toku toriaezu
    //ここちゃんと表示できるように
    if(!a){
        printf("=== ASSERT [%s] ========> PROGRAM ETREMINATED\n" , comment);
        //一応メモリプール解放しまうｓ
        exit(-1);
    }
}


//CPU,GPUの種類
enum {
    CPU,GPU
};

//--------------------------------------------------------------------------------------------------------------
//値と勾配のセットを定義しておきます。
//--------------------------------------------------------------------------------------------------------------
struct weight{
    double v;        //a
    double d;        //勾配
    __host__ weight() :v(0.0), d(0.0) { ; }  //一応初期化書いてみる。
};

//CPUとGPU両方に対応したアドレスを表す型を作っておきます。
template<typename T>
class cpu_gpu_mem
{
public:
    T* cpu;
    T* gpu;
    int size;
    cpu_gpu_mem() : cpu(nullptr),gpu(nullptr){;}
    cpu_gpu_mem(int sz) :cpu_gpu_mem() {
        alloc(sz);
    }
//    cpu_gpu_mem(T* _cpu, T* _gpu) : cpu(_cpu), gpu(_gpu) { ; }
    cpu_gpu_mem(const cpu_gpu_mem<T>&a) = default;

    template<typename S> operator cpu_gpu_mem<S> () const {
        return cpu_gpu_mem<S>((S*)cpu, (S*)gpu);
    }
    cpu_gpu_mem<T>& operator=(const cpu_gpu_mem<T>& a) = default;
    //
    __host__ cpu_gpu_mem& alloc(int sz)
    {
        _Assert((cpu == nullptr) && (gpu == nullptr), "already memory allocated");
        //CPU
        try {
            cpu = new T[sz];       //※デフォルトコンストラクタで作ります。
        }
        catch (const bad_alloc& e) { printf("new[] failed(%s)\n", e.what()); _Assert(false, "cpu_gpu_mem : alloc failed "); }
        //GPU
        cudaError_t e;
        if ((e = cudaMalloc((void**)&gpu, sz * sizeof(T))) != cudaSuccess) { _Assert(false, "cpugpu_alloc : cudaMalloc failed"); }
        size = sz;
        return *this;
    }
    //cpu→gpuへ転送する。
    __host__ void Transfer_to_GPU() {
        //ここでヌルポインタをチェックします。
        if (!cpu && !gpu) {
            return;//
        }
        _Assert(cpu && gpu, "Tranfser_to_GPU nullpointer");
        cudaError_t cudaStatus = cudaMemcpy((void*)gpu, (void*)cpu, size * sizeof(T), cudaMemcpyHostToDevice); //GPUへ転送する。
        _Assert(cudaStatus == cudaSuccess, "Transfer_to_GPU() error");
    }
    __host__ void Transfer_to_CPU() {
        if (!cpu && !gpu) {
            return;//
        }
        _Assert(cpu && gpu, "Tranfser_to_CPU nullpointer");
        cudaError_t cudaStatus = cudaMemcpy((void*)cpu, (void*)gpu, size * sizeof(T), cudaMemcpyDeviceToHost); //GPUへ転送する。
        _Assert(cudaStatus == cudaSuccess, "Transfer_to_CPU() error");
    }
    __host__ void free()
    {
        if (cpu != nullptr) {
            delete cpu;
        }
        if (gpu != nullptr) {
            cudaFree((void*)gpu);
        }
    }
    __host__ __device__ void dump()const{printf("CPU[%p]GPU[%p]size[%d]\n",cpu, gpu , size); }

    //()でアドレスを取得できると読みやすくなるか
    __host__ __device__ T& operator()(int location,int  idx)const{
        _Assert(idx < size , "cpu_gpu_mem::() size illegal");
        if(location == GPU){
            return gpu[idx];
        }
        return cpu[idx];
    }
    __host__ __device__ T& operator()(int location=CPU) {  //なにも添え字を指示しない場合は0番目を
        return operator()(location, 0);
    }


};



//エスケープシーケンス
__host__ __device__ void _pos(int x, int y);
__host__ __device__ void esc_clr();

//gpuでは可変個引数を取れない
//__host__ void pos_printf(int x, int y, char* fmt, ...);

#endif  //

/*
//------------------------------------------------------------------------------------------------------------------------------------
//  各パーセプトロンから、要求されてメモリのインデックスを返します。
//  現在、どこにそのくらい割付をしたかは管理しない。オーバーしてしまった場合だけひっかけます。あとは自己責任でやるようにします。
//------------------------------------------------------------------------------------------------------------------------------------
__host__ cpu_gpu_adr<T> alloc(int sz)
{
    _Assert(_allocated + sz <= _size, "alloc overflow");   //_sizeまでアロケートできる。
    cpu_gpu_adr<T> a;      //戻りを作っていきます。
    a.cpu = _adr.cpu + _allocated;        //確保している範囲を超えるときはnullにします。(cpu側には確保できなかった)
    a.gpu = _adr.gpu + _allocated;       //                          同上
    _allocated += sz;
    return a;						//ホスト側ハンドル（実際には全体のカウンタです）
}
*/
//------------------------------------------------------------------------------------------------------------------------------------
//  指示された場所のポインタを返します。条件分岐しないように重複させます。
//------------------------------------------------------------------------------------------------------------------------------------    
/*    __host__ T* cpu(int h=0, int idx=0) {
        _Assert(_cpu!=nullptr , "cpu_gpu_memory::cpu() : cpu memory is not allocated" );
        _Assert( h + idx < sizs , "cpu() : h +idx overflow");
        return _cpu + (h + idx);
    }
    __device__ T* gpu(int h = 0, int idx = 0) {
        _Assert(_cpu!=nullptr , "cpu_gpu_memory::gpu() : cpu memory is not allocated" );
        _Assert(h + idx < sizs, "cpu_gpu_memory::gpu() : h +idx overflow");
        return _gpu + (h + idx);
    }
*/

//エスケープシーケンス

__host__
void dumpGPUInfo();
