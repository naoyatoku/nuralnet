
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"


struct node
{
    float a;                // 入力
    float dE_da;            //
    float y;                //    
    cpu_gpu_mem<float>w;    //全段nodeの重みづけです
    float b;               //バイアス項です。


    //最終段はソフトマックスにしないと
    __device__ float activate() {  // ReLU限定にしてみる
        return (a < 0.0f) ? 0.0f : a;
    }

    __device__ float d_activate() {
        return (a > 0.0f) ? 1.0f : 0.0f;
    }
    __host__ node() : a(0.0), dE_da(0.0), y(0.0),b(0.0) { 
        a=0.0;
        ; 
    }


    //重みづけ
    //matrixを持つようにする。
    //このノードに対する前段ノード数分の重み（行列の１行分）現段ノード数分の行が行がある。行列となる。
    //
    //  
    //バックプロパゲーションのときにwのupdate1もこのほうがわかりやすいんじゃないか。
    __host__ void alloc_w(int n_prev_nodes) {
        w.alloc(n_prev_nodes);
        __w_init_std();             //適当な値で初期化します。
    }
    //重みに適当な値をセットする。
    __host__ void __w_init_Xavier(void);
    __host__ void __w_init_He(void);
    __host__ void __w_init_std(void);

    //cpu<->gpuメモリ転送
    __host__ void Transfer_to_GPU() {
        w.Transfer_to_GPU();
    }
    __host__ void Transfer_to_CPU() {
        w.Transfer_to_CPU();
    }
    //dump
    __host__ void dump(int l,int n)const {
//        esc_clr();
        pos_printf( l*40 , n+1 , "a[%8.6f]y[%8.6f]dEda[%8.6f]",a,y,dE_da );
    }

};