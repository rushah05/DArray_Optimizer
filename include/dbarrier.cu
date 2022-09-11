// #pragma once
#include <cassert>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <string>


cublasHandle_t handle;
cusolverDnHandle_t csHandle;
cudaStream_t stream;
cudaError_t cudaStat;
cublasStatus_t stat;
cusolverStatus_t statusH = CUSOLVER_STATUS_SUCCESS;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void cuda_init(){
    cublasCreate(&handle);
    cusolverDnCreate(&csHandle);
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    cusolverDnSetStream(csHandle, stream);
}

void cuda_finalize(){
}

template<typename T>
void printMatrixDeviceBlock(const std::string filepath, int m, int n, T* dA, int lda){
    FILE *f = fopen(filepath.c_str(), "w");
    assert(f);

    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++){
        for(int j = 0;j<n;j++){
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, " ,");
        }
    }
    fclose(f);
    free(ha);
}


template<typename T>
void printVectorDeviceBlock(const std::string filepath, int m, T* dA){
    FILE *f = fopen(filepath.c_str(), "w");
    assert(f);

    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++){
        cudaMemcpy(&ha[0], &dA[i], sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(f, "%lf", ha[0]);
        fprintf(f, "\n");
    }
    fclose(f);
    free(ha);
}


void logbarrier(double* K, int ldk, int m, int n){
    double *dK, *da;
    gpuErrchk(cudaMalloc(&dK, sizeof(float)*m*n));
    gpuErrchk(cudaMemcpy(dK, K, sizeof(float)*m*n, cudaMemcpyHostToDevice));
       
    gpuErrchk(cudaMalloc(&da, sizeof(float)*m));
    
}


































// template<typename T>
// T f(int rk, int n, int k, DArray::LMatrix<T> x, DArray::LMatrix<T> G){
//     // f(x) = (x'*(G*(G'*x))) + sum(x)
//     T done = 1, dzero = 0;
//     int one=1;
//     DArray::LMatrix<T> Gx(k, 1);
//     // printf("G[%d,%d] x[%d,%d] Gd[%d,%d] \n", G.dims()[0], G.dims()[1], x.dims()[0], x.dims()[1], Gx.dims()[0], Gx.dims()[1]);
//     dgemm_("T", "N", &k, &one, &n, &done, G.data(), &G.ld(), x.data(), &x.ld(), &dzero, Gx.data(), &Gx.ld()); 
    
//     DArray::LMatrix<T> GGx(n, 1);
//     dgemm_("N", "N", &n, &one, &k, &done, G.data(), &G.ld(), Gx.data(), &Gx.ld(), &dzero, GGx.data(), &GGx.ld());
    
//     T xGGx = 0;
//     // if(rk==0) printf("x[%d,%d], GGx[%d,%d]\n", x.dims()[0], x.dims()[1], GGx.dims()[0], GGx.dims()[1]);
//     dgemm_("T", "N", &one, &one, &n, &done, x.data(), &x.ld(), GGx.data(), &GGx.ld(), &dzero, &xGGx, &one);
    
//     T xsum = 0;
//     for(int i=0; i<n; i++){
//         xsum+=x.data()[i];
//     }

//     // if(rk == 0) printf("xGGx::%f, xsum::%f, fval::%f\n", xGGx, xsum, xGGx-xsum);
//     return xGGx-xsum;
// }


// template<typename T>
// T g(int rk, int n, int k, DArray::LMatrix<T> x, DArray::LMatrix<T> G, double C, int t){
//     T gval = 0;
//     for(int i=0; i<n; ++i){
//         if(x.data()[i]<0.0 || x.data()[i]>C){
//             gval=-1;
//             break;
//         }
//     }
//     if(gval==0){
//         T fval = f(rk, n, k, x, G);
//         T xsum = 0;
//         for(int i=0; i<n; i++){
//             xsum+=(log(x.data()[i])+log(C-x.data()[i]))/t;
//         }
//         gval=fval-xsum;
//     }
//     return gval;
// }







