#include "common.h"
#include "node.h"
#include <stdarg.h>

struct layer
{
    cpu_gpu_mem<node> nodes;  //
    __host__ layer() { ; }
    __host__ void alloc_nodes(int n_nodes ,layer*p_prev_layer =0 ) {
        nodes.alloc(n_nodes);
        //一番前はｗを設定しない。
        if (p_prev_layer) {
            for (int i = 0; i < nodes.size; ++i) {
                nodes.cpu[i].alloc_w(p_prev_layer->nodes.size);
            }
        }
    }
    __host__ void Transfer_to_GPU() {
        for (int i = 0; i < nodes.size; ++i) {      //nodeのメンバをメモリ転送する指示です。
            nodes.cpu[i].Transfer_to_GPU();
        }
        nodes.Transfer_to_GPU();                    //node配列自身のメモリ転送です。
    }
    __host__ void Transfer_to_CPU() {
        for (int i = 0; i < nodes.size; ++i) {
            nodes.cpu[i].Transfer_to_CPU();         
        }
        nodes.Transfer_to_CPU();
    }
    __host__ void dump(int l)const {
        for (int i=0; i < nodes.size; ++i) {
            nodes.cpu[i].dump(l,i);
        }
    }

};

enum {
	LOSS_MEAN_SQUARE,
	LOSS_ENTROPY,
	LOSS_HUBER,
    LOSS_SOFTMAX_WITH_CROSSENTROPY , //
    LOSS_NOP,                      //なにもしない
};
template<size_t N>
struct net
{
    int n_layers;
    layer layers[N];  //N層レイヤーです。

    __host__ net()  {   _Assert(0,"net : default constructor is not available"); }  // デフォルトコンストラクタ
    __host__ net(int n_nodes_1 ,  ...)  :   n_layers(N)
    {
        layers[0].alloc_nodes(n_nodes_1);       //1番目
        va_list ap; va_start(ap,n_nodes_1);     //2番目以降は可変個処理します。
        for (int i = 1; i < N; ++i ) {
            layers[i].alloc_nodes(
                va_arg(ap, int)
                , &layers[i-1] );
        }
    }
    //このオブジェクト自体をGPUに転送して、実行すると、nodes,nodes::__wがcpu_gpu_mem両方のアドレスを持っているため、
    //netオブジェクト自体は変化がないので再度転送する必要ない。
    //nodeと、それにともなう__wをGPU→CPUへ転送するだけでよい。
    __host__ net<N>* Transfer_to_GPU() {
        for (int l = 0; l < n_layers; ++l) {
            layers[l].Transfer_to_GPU();
        }
        //自分自身をGPUへ転送します。
        //netオブジェクト自体を転送する。
        net<N>* pnet;
        {
            cudaError_t e;
            //alloc
            if ((e = cudaMalloc((void**)&pnet, sizeof(net<N>))) != cudaSuccess) { _Assert(false, "cpugpu_alloc : cudaMalloc failed"); }
            //copy
            cudaMemcpy((void*)pnet, (void*)this, sizeof(net<N>), cudaMemcpyHostToDevice);
        }
        return pnet;
    }
    __host__ void Transfer_to_CPU() {
        for (int l = 0; l < n_layers ; ++l) {
            layers[l].Transfer_to_CPU();
        }
        //net自体は、GPU上でのものと変わりはないのでコピーしない。（GPUの先で必要になったらコピーする。）
    }
    __host__ void dump()const {
        esc_clr();
        for (int l = 0; l < n_layers; ++l) {
            layers[l].dump(l);
//            printf("\r\n");
        }
    }

    //入力と、正解の組み合わせをセットして、学習を行えるようにする。
    __host__ void set_input_answer(float *in,int in_size , float *answer,int ans_size)
    {
        //入力は、入力段コピーするようにします。
        _Assert(in_size == layers[0].size , "net::set_input_answer : in_size illegale(%f,%f)" , in_size , layers[0].size );
        for(int i=0 ; i < in_size ; ++i){
            layers[0].nodes.cpu[i].a = in[i];
        }
        //answerは

    }
    //最終レイヤーのdE/daを求めるための
    __device__ void loss_softmax_with_crossentropy();


};

