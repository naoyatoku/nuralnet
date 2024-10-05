#include "cuda_runtime.h"
#include "net.cuh"

#include <stdarg.h>

//----------------------------------------------------------------------
//      constructor
//----------------------------------------------------------------------
 __host__ void  net::construct(int n_layers , int n_input ,int n_mid_nodes , int act_type , int n_output_nodes, int output_act_type , int loss_type)
{
    //レイヤーの割付です。
    ly.alloc(n_layers);
    //各レイヤーのノードを割り付けていきます。
    {
        ly(CPU,0).alloc_nodes(0,n_input);                           //1番目
        for (int i = 1; i < ly.size-1; ++i ) {                        //隠れそう
            ly(CPU,i).alloc_nodes(i,n_mid_nodes , act_type, &ly(CPU,i-1) );
        }
        ly(CPU, n_layers - 1).alloc_nodes(n_layers - 1, n_output_nodes, output_act_type , &ly(CPU,n_layers -2 ) );       //出力層
    }
}
//----------------------------------------------------------------------
//      CPU--->GPU転送
//----------------------------------------------------------------------
__host__ void net::Transfer_contained_members_to_GPU()
{
    //nodeの転送です。
    for (int l = 0; l < ly.size; ++l) {
        ly(CPU,l).Transfer_contained_members_to_GPU();
    }
    //layerの転送です。
    ly.Transfer_to_GPU();
}
//----------------------------------------------------------------------
//      CPU<---GPU  転送
//----------------------------------------------------------------------
__host__ void net::Transfer_contained_members_to_CPU() 
{
    //net自体を転送します。
    for (int l = 0; l < ly.size ; ++l) {
        ly(CPU,l).Transfer_contained_members_to_CPU();
    }
    //net自体は、GPU上でのものと変わりはないのでコピーしない。（GPUの先で必要になったらコピーする。）
}
__host__ __device__ 
void net::dump(int locate)const
{
    esc_clr();
    for (int l = 0; l < ly.size; ++l) {
        ly(locate,l).dump(l,locate);
//            printf("\r\n");
    }
}

//入力と、正解の組み合わせをセットして、学習を行えるようにする。
__host__ void net::set_input_answer(float *in,int in_size , float *answer,int ans_size)
{
    //入力は、入力段コピーするようにします。
    _Assert(in_size ==  ly(CPU,0).nodes.size , "net::set_input_answer : in_size illegale(%f,%f)" , in_size , ly(CPU,0).nodes.size );
    for(int i=0 ; i < in_size ; ++i){
        ly(CPU, 0).nodes(CPU, i).a = in[i];
    }
}