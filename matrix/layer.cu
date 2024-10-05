#include "layer.cuh"


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

#include <stdarg.h>
#include "kernels.cuh"
__device__
float layer::request(int cmd,...)const
{
    va_list ap;va_start(ap,cmd);
    switch(cmd){
        case REQ_CALC_MAX:
            calc_max( nodes.gpu , nodes.size);  //とりあえず呼べるかどうか。
            return 0.1;
        case 1:
            return 1.0;
    }
    va_end(ap);
    return 2.0;
}