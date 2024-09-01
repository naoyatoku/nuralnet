
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <random>
#include <iostream>
#include <vector>
using namespace std;
/*
__global__ void matrix_2d_kernel(float* w , float* M , float* N ) {
    //blockは一個(blockIdx=0)
    //これで1blockあたりx:m y:n 個のスレッドがある 
    // row_idx  = threadIdx.x 
    // idx = threadIdx.y

    atomicAdd( N + threadIdx.y , w[threadIdx.x + ( threadIdx.y*blockDim.x) ] * M[threadIdx.x] );
    printf("x[%d]y[%d] : w[%d][%d]=%f  M[%d]:%f  -> N[%d]:%f \n"
        , threadIdx.x, threadIdx.y
        , threadIdx.y, threadIdx.x, w[threadIdx.x + (threadIdx.y * blockDim.x)]
        , threadIdx.x, M[threadIdx.x]
        , threadIdx.y, N[threadIdx.y]
    );
// 
//    N[0] = 0.1;
}


__host__ void initialize(float* d, int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // 配列wの初期化
    for (int i = 0; i < size; ++i) {d[i] = dis(gen);}

    // 結果を表示（オプション）
    for (int i = 0; i < size; ++i) { printf("%d[%lf]\n", i, d[i]); }
}


*/


//重みづけマトリクスは別に?
//レイヤー間で、
//  (前段ノードの数)
//各レイヤーごとのノード数マトリクスです。これは、netを作るときに数が行列の数が決まるので
//float *w[MAX_LAYER];   //これがレイヤーの間にある。ー＞レイヤーの数分あるこれらは全部動的に作ってみる。
//継承を使わないノードを作る
struct node
{
    float   a;      //入力
    float   da_de;  //
    float   y;      //


    __device__  float activate(){       //relu限定にしてみる。
        if(a<0)return 0.0;
        return a;
    }
    __device__  float d_activate()
    {
        if(a>0.0){  return 1.0;}
        return 0.0;       //
    }
    //
};
template<size_t _n>
struct layer
{
    int n;
    node nodes[_n];
    layer():n(_n){;}
};

//継承しないnet
template<size_t _n>
struct net
{
    layer*layers;
    int n_layer;
    net(int l):n_layer(l){;}
};

//なんか
void do_net()
{
    int layers=0;       //これがnetかな？
    //構成するネットを作ります。
    int nodes[] = {16,8,36,-1};     //とりあえず固定で
    const   int layers = 3;         //３レイヤー

//    net _net<3>;
}





 
int main() {

/*
    //n次元ベクトル→m次元ベクトルへ変換する線形写像行列計算、

    const int m = 4;       //今段レイヤー数(ベクトル次元)
    const int n = 5;        //次段レイヤー数(ベクトル次元)

    //  ベクトルを縦にすると、
    // ベクトルは（m*n）
    // 
    //  行列A() * ベクトルm => ベクトル n
    //

    //重みづけを定義します。
    float w[n][m]={ {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1},{1,0,0,1}};         //m列がn行で縦のベクトル[n]ができる
    float M[m] = { 1,2,3,4 }, N[n] = {0,0,0,0,0};              //各ベクトルです。
    //0-1の間で初期化
    initialize(&w[0][0],n*m);   //
//    initialize(&M[0],m);
//    initialize(&N[0],n);

    float* w_dev;            //デバイス側のメモリポインタ
    float* M_dev,*N_dev;      //
    
    // デバイスメモリを確保
    {
        cudaMalloc((void**)&w_dev, sizeof(w));
        cudaMalloc((void**)&M_dev, sizeof(M));
        cudaMalloc((void**)&N_dev, sizeof(N));
        cudaMemcpy( w_dev , &w[0][0]    , sizeof(w), cudaMemcpyHostToDevice);
        cudaMemcpy( M_dev , &M[0]       , sizeof(M), cudaMemcpyHostToDevice);
        cudaMemcpy( N_dev , &N[0]       , sizeof(N), cudaMemcpyHostToDevice);
    }

    // カーネルを起動
//    kernel<<blocks_per_grid,threads_per_block>>()
//    dim3 threads_per_block(n,m);
   // dim3 blocks(1, 0);
 //  kernel<<< 1 ,threads_per_block >>>(w_dev,M_dev,N_dev);
//    kernel<<< 1 ,256 >>>(w_dev,M_dev,N_dev);

    dim3 threads_per_block(m, n);
    matrix_2d_kernel << <1, threads_per_block >> > (w_dev, M_dev, N_dev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();


    // 結果をホストにコピー
    cudaMemcpy( (void*)&N[0] , N_dev,  sizeof(N), cudaMemcpyDeviceToHost);


    for (int i = 0; i < n; ++i) {
        printf("%f\n" , N[i]);
    }
//    printf("Result: %d\n", hostData[0]);

    // デバイスメモリを解放
    cudaFree(w_dev);
    cudaFree(M_dev);
    cudaFree(N_dev);
*/
    return 0;
}

//
//    N[0]   = M[0]*w[0][0] + M[1]* w[0][1] + ....  M[m]*w[0][m-1]
  //  N[1]   = M[0]*w[1][0] ....................... M[m]*w[1][m-1]
  //                             :
  //  N[n-1] = M[0]*w[n-1][0] + ..................  M[m]*w[n-1][m-1]


    //行列でやってみる。
    //ニューラルネットの計算は、重みづけが
    //横が次レイヤーノード数 n
    //縦が今レイヤーノード数 m
#if 0
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

//-----------------------------------------------------------------------------
// backwardの計算
//-----------------------------------------------------------------------------


損失レイヤーのdE/da計算
virtual double activate(void) {
    //ここで本来重みづけは、w[0,0,1,0,0,0,]というようなn番目が1の配列だが、結果n番目の入力そのものなのでそれを採用するようにします。
//		_a		=	in->at(_n)->out();		//前段の出力をそのまま使います。
        //損失関数を呼びます。
    _Assert(_in != nullptr, "error_perceptron::activate() : ");      //_この時点でinが確定している必要があります。(affine呼び出しで登録される。)
    _out = _loss->E(_a, _t, this);
    損失関数で出力を決める。

    //ここでこのオブジェクトの勾配も求まります。
    _dE_da = _loss->dE_dy(_a, _t);

    //
    dE_dy()は損失関数の結果です。

    return _out;
}


//損失レイヤーでは損失関数によりdE/daが計算されている。


//損失レイヤーの一つ手前から始める。
for (int l = 損失レイヤの一つ手前; 最初のレイヤーまで; l--)
{
    kernel
    for (lのノードサイズ) {
        double de_dy = 0; //今回の更新
        //kernel
        for (l + 1のノードサイズ) {
            de_dy[i]  +=  net[l + 1]->node[j]->dE_da() * net[l + 1]->node[j]->_w[i];        //Σの部分
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            de_dy[i]   =  Σ      次段ノード[j] のdE / da           ×      次段のノード[j] の  w[i]
            ^^^^^
           ※ ノード[i]のde_dy(出力結果の損失度合)
        }

        net[l]->node[i]->set_dE_da(de_dy* net[l]->node[i]->d_act());        //σの部分
        ※ノード[i]のdE_da =  de_dy[i] * d_act[i]アクティベータの微分


        ｗ更新_wのサイズは前段ノードの数です。
        //
        for (int x = 0; x < net[l]->node[i]->_w.size(); ++x) {      //
            net[l]->node[i]->_w[x] -= _lr * net[l]->node[i]->dE_da() * net[l]->node[i]->in(x);
                //^^^^^^^^^^^^^^^^^^^^ da[L]i/dw[L]xi => a[L]iへの前段[L-1]のnode[x]の出力	( = net[l]->node[i]->in(x) )
                ※ w[x]  -= （学習率） *  dE/da[i] * 入力値(i)<- (x)
         }
    }

}
//各ノード
//ベクトルと、次段レイヤーの全ノードに対する





void backward(vector<layer*>& net) {        //layer：forwardを行ったネット。vector<double>tは、
    //layersには、入力層[0]+中間層[1]-[N]+損失関数のあるレイヤー[N+1]でなっている。更新する必要があるのは、[1]-[N]の層のw。
    //layaer.size()は、損失関数のレイヤも含む. 更新する必要がある層の添え字は、[size()-2]->[1]
    for (int l = net.size() - 2; l > 0; --l) {       //レイヤーの数分です。（入力レイヤーは除く）
        for (int i = 0; i < net[l]->node.size(); ++i) {                 //1つのレイヤーの各ノードに対して最適化していきます。
            //まずdE/da(L)[i]を作ります。一番最後のレイヤーから実行する前提で書いています。（はじめからやるとうまくいきません。)
            {
                double de_dy = 0; //今回の更新
                for (int j = 0; j < net[l + 1]->node.size(); ++j) {
                    de_dy += net[l + 1]->node[j]->dE_da() * net[l + 1]->node[j]->_w[i];        //Σの部分
                }
                net[l]->node[i]->set_dE_da(de_dy * net[l]->node[i]->d_act());        //σの部分
            }
            //dE/da(L)iが決まったので、wを更新していきます。
            for (int x = 0; x < net[l]->node[i]->_w.size(); ++x) {      //
                net[l]->node[i]->_w[x] -= _lr * net[l]->node[i]->dE_da() * net[l]->node[i]->in(x);
                //^^^^^^^^^^^^^^^^^^^^ da[L]i/dw[L]xi => a[L]iへの前段[L-1]のnode[x]の出力	( = net[l]->node[i]->in(x) )
            }
            //バイアスを更新です。
            net[l]->node[i]->_b -= _lr * net[l]->node[i]->dE_da();
        }
    }
}


#endif
