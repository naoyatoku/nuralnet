#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "net.h"

/*
//gpu用に損失関数と、その各ノードのdE/daを求めるように
__device__ loss_E_softmax_with_crossentropy(float y, float t , float &softmax_y)
{
    float sum = 0.0;
    {
            float max = -100000.0;	//マイナスの大きな値にしておきます。
            //まずmaxを計算します。
            //
            //tokuこれがだめ。pは、自分のノード。
            //同じ列の入力を見ないといけないので、in_array()のout()を見る必要があります。
            //affineの後なので、本来、同レイヤーの
            for (int i = 0; i < in_size() ; ++i) {
                if ( in_array[i].out() > max) { max = in_array[i].out(); }
            }	//maxを算出していきます。
            for (int i = 0; i < in_size() ; ++i) {
                sum += exp( in_array[i].out() - max);
//                if (_idx == 0) {
//                    printf("  softmax: in[%d] : %lf \n" , i , in_array[i].out());
//                }
            }	//これはaffin後の入力です。


            //今回の活性化の値は、以下です。
            //		_Assert(sum!=0.0 , "softmax::act() : sum is zero");
            if (sum == 0.0)sum = 0.00000001;	//0だった場合にはとても小さな値にしてみる。
            softmax_y = exp( a() - max) / sum;//ソフトマックスと、クロスエントロピー誤差を組み合わせた損失関数
//            printf("softmax[%d]=%lf   max=%lf , sum=%lf , in.out:%lf  a:%lf\n " , _idx , softmax_y , max ,sum , in_array[_idx].out() , _a);
            //^^^^^^
    }
}
__device__ float loss_dE_dy_softmax_with_crossentropy(double y, double t)
{
        //クロスエントロピーと、ソフトマックスが全結合なし（一対一でつながっている）とき、
        //  
        //　この層の

        //ちょっとあいまい
        //  dE           dE                da(cross_entro)      
        // ---- =   -----------------  * ------------------
        //  da           dy                da(softmax)

        //クロスエントロピー誤差の微分は、 -t / y ; -t*log(y)
        //	{  y'  =  -1 * (t / (y + EPS));   }
            //さらに、ソフトマックスの微分
        //	{ y * (1 - y);						}
//        printf("dEdy softmax[%d] => %lf  (y:%lf  t:%lf) \n", _idx, y - t ,y ,t );
        return y - t;       //※yに、ソフトマックスの結果を入れる。
//        return softmax_y - t;
}
*/



//ソフトマックスは
//
//              exp(an - amax)
//      ----------------------------------------
//          Σ   exp( ai  - amax)
//          i
//
template<size_t N>
__device__ void net<N>::loss_softmax_with_crossentropy()
{
    layer &l  = layers[N-1];    //最終レイヤー
    //GPUを利用して、MAXを計算します。
   int x = threadIdx.x;


   //一番近い2のs累乗の数字を探す。
   {
       int _n = 1;
        while (_n < N) {
            _n <<= 1;  // 2倍していく
        }
        //ここで
   }

   float max[N];    //
   //奇数だったらどうなるか
   for(int n=N/2 ; n>1 ; n/=2 ){                                //N=7 : n=3         n=1
        if(x < n){                                              //thread.xは0,1,2 
            max[x] = l.nondes.gpu[x] > l.nodes.gpu[x+n];        //max(a[0],a[3])   , max(a[1],a[4]) , max[[2] ,[5] ,   
                                                                //      max[0]          max[1]          max[2]
                                                                // 
        }
        __syncthreads();
   }
   //ここでmax[0]が最大値になっています。
   //奇数の場合、max[0]
   if(N%2!=0){

   }




}



#if 0
template<size_t N>
__device__ void net<N>::loss_softmax_with_crossentropy()
{
    //最終レイヤーに対して処理を施します。

    //gpu上での
    int node = threadIdx.x;
    //最終レイヤーのノードに対してのみ行う。
    if(node > layers[n_layer].size){        //
        goto _calc_dEda;
    }

    float cross_entropy_loss;           //損失関数です。
    {
        //    double  softmax_y;

        float softmax;

        //ソフトマックスを計算するときは、最終レイヤの 最大値と、合計を計算する必要がある。


        
    	float sum = 0.0;
        {
            va_list ap;	va_start( ap , t );						//第一引数に perceptron*
            const perceptron*p = va_arg(ap,const perceptron*);
    	    va_end(ap);
            double max = -100000.0;	//マイナスの大きな値にしておきます。
		    //まずmaxを計算します。
		    //
		    //tokuこれがだめ。pは、自分のノード。
		    //同じ列の入力を見ないといけないので、in_array()のout()を見る必要があります。
		    //affineの後なので、本来、同レイヤーの
    		for (int i = 0; i < p->in_array()->size(); ++i ) {		if(p->in_array()->at(i)->out() > max)		{ max = p->in_array()->at(i)->out(); }	}	//maxを算出していきます。
	    	for (int i = 0; i < p->in_array()->size(); ++i ) {
		    	sum += exp( p->in_array()->at(i)->out() - max);
		    }	//これはaffin後の入力です。
		    //今回の活性化の値は、以下です。
            //		_Assert(sum!=0.0 , "softmax::act() : sum is zero");
		    if(sum==0.0)sum=0.00000001;	//0だった場合にはとても小さな値にしてみる。
		    softmax_y = 	exp(p->a()-max) / sum;//ソフトマックスと、クロスエントロピー誤差を組み合わせた損失関数
    	    //^^^^^^
        }
        cross_entropy_y = -1.0 * t * log(softmax_y);
    }
_calc_dEda:
    ;

}
#endif