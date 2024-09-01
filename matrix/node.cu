#include "common.h"
#include <random>
#include "node.h"

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
