#include "common.cuh"
#include <random>
#include "node.cuh"

using namespace std;
#include<stdio.h>

//乱数を発生させるためのシードです。
static random_device rd;
static mt19937 gen(rd());
//static std::normal_distribution<float> distribution(0.0,1.0);		//標準偏差で初期化する。

//    cpu_gpu_adr<weight> __w;             //重みづけメモリのポインタを持ってみます。これはGPUだったり、CPUだったりする。

//Xavier初期化
__host__ void node::__w_init_Xavier(void){
	_Assert(  w.size > 0, "w_init_xavier() : in size is 0");
//	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, std::sqrt(1.0 / w.size ));
    for(int i=0 ; i < w.size ; ++i ){   w.cpu[i] = distribution(gen); }
}
//He初期化
__host__ void node::__w_init_He(void){
//	int fan_in = _w.size();					//前段のノードの数は、_wのサイズに割り当てられています。
//	std::default_random_engine generator;
	_Assert(w.size > 0, "w_init_He() : in size is 0");
	std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / w.size ));
    for(int i=0 ; i < w.size ; ++i ){  
		w.cpu[i] = distribution(gen);
	}
}
//正規分布による初期化
__host__ void node::__w_init_std(void){
	_Assert( w.size > 0, "w_init_std() : in size is 0");
	normal_distribution<> dist(0.0,.01);            //
    for(int i=0 ; i < w.size ; ++i ){
		w.cpu[i] = dist(gen);        
		printf("%lf,", w.cpu[i]);
	}
	printf("\r\n");
}
__device__ float node::activate(const layer *parent)
{  // ReLU限定にしてみる
    switch(act_type){
    case ACT_SOFTMAX:
        return activate_softmax(parent);
        break;
     case  ACT_RELU:
         return (a < 0.0f) ? 0.0f : a;
     case ACT_NOP:
         return a;
         break;
    }
}

__device__ float node::d_activate()
{
    switch (act_type) {
    case ACT_SOFTMAX:
        return d_activate_softmax();
        break;
    case ACT_RELU:
        return (a > 0.0f) ? 1.0f : 0.0f;
    case ACT_NOP:
    default:
        return 1.0f;
        break;
    }
}
__host__ void node::alloc_w(int n_prev_nodes)
{
    w.alloc(n_prev_nodes);
    __w_init_std();             //適当な値で初期化します。
}
//内包メンバのGPU転送です。
__host__ void node::Transfer_contained_members_to_GPU() 
{
    w.Transfer_to_GPU();
}
__host__ void node::Transfer_contained_members_to_CPU() {
    w.Transfer_to_CPU();    
}
__host__ __device__
void node::dump(int l,int n)const 
{
    _pos(l * 40, n + 1);
    printf(  "a[%8.6f]y[%8.6f]dEda[%8.6f]",a,y,dE_da );
}

//ノードののアクティベーションはmaxを計算するためにnetに依存する。が、すごくいやだ。

#include "layer.cuh"    //ここでレイヤーの情報を得るのはいいらしい。
__device__ float node::activate_softmax(const layer*p_parent_layer)
{
//    const layer& _l = _net->layers[this->nlyr];        //自分が所属するレイヤーです。
//    float max;
    if (threadIdx.x == 0) {
        p_parent_layer->request(REQ_CALC_MAX);

//        parent.request(0);
//            max_kernel <<< 16, 1 >>> (0);
    }
//    __syncthreads();
    return 0.0;
}
__device__ float node::d_activate_softmax(){            //
    return 0.0;
}
