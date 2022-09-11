// #pragma once
#include <cassert>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <string>

float *dXi, *dXj, *dXij, *dYi, *dYj, *dK, *dA, *dB, *dC;
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
    // cudaMalloc(&mem, sizeof(float)*65536*65536);
	// cudaMalloc(&half_mem, sizeof(__half)*65536*65536);
    cublasCreate(&handle);
    cusolverDnCreate(&csHandle);
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    cusolverDnSetStream(csHandle, stream);
    // int bs=8192;
    int bs=16384;
    gpuErrchk(cudaMalloc(&dXi, sizeof(float)*bs));
    gpuErrchk(cudaMalloc(&dXj, sizeof(float)*bs));
    gpuErrchk(cudaMalloc(&dXij, sizeof(float)*bs*bs));
    gpuErrchk(cudaMalloc(&dYi, sizeof(float)*bs));
    gpuErrchk(cudaMalloc(&dYj, sizeof(float)*bs));
    gpuErrchk(cudaMalloc(&dK, sizeof(float)*bs*bs));
    gpuErrchk(cudaMalloc(&dA, sizeof(float)*bs*bs));
    gpuErrchk(cudaMalloc(&dB, sizeof(float)*bs*bs));
    gpuErrchk(cudaMalloc(&dC, sizeof(float)*bs*bs));
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


__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

template<typename T>
__global__ 
void vecnorm(T *Zd, int ldz, T *ZI, int m, int k){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if( i<m ) {
		T sum = 0.0f;

        #pragma unroll (4)
		for( int j=0; j<k; j++ )
			sum += Zd[i+j*ldz]*Zd[i+j*ldz];

		ZI[i] = sum;
	}
}

__global__
void rbf( int m, int n, float *buf, int ldb, float *XI, float *XJ, float *XIJ, int ldxij, float gamma, float *YI, float *YJ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if (i<m && j<n) {   
        buf[i+j*ldb] = YI[i]*YJ[j]*__expf(-gamma*(XI[i]+XJ[j]-2*XIJ[i+j*ldxij]));
    //    printf("[i,j]=[%d,%d], buf[]=%.4f, XI[]=%.4f, XJ[]=%.4f, XIJ[]=%.4f\n", i, j, buf[i+j*ldb],
	// 		   XI[i], XJ[j], XIJ[i+j*ldxij]);
	}
}

void Kernel(int rk, int m, int n, float *Xi, float *Xj, float *Yi, float *Yj, float *Xij, int ldxij, float *K, int ldk, float gamma){
    // float *dXi, *dXj, *dXij, *dYi, *dYj, *dK;
    // gpuErrchk(cudaMalloc(&dXi, sizeof(float)*m));
    // gpuErrchk(cudaMalloc(&dXj, sizeof(float)*n));
    // gpuErrchk(cudaMalloc(&dXij, sizeof(float)*m*n));
    // gpuErrchk(cudaMalloc(&dYi, sizeof(float)*m));
    // gpuErrchk(cudaMalloc(&dYj, sizeof(float)*n));
    // gpuErrchk(cudaMalloc(&dK, sizeof(float)*m*n));
    gpuErrchk(cudaMemcpy(dXi, Xi, sizeof(float)*m, cudaMemcpyHostToDevice));  
    gpuErrchk(cudaMemcpy(dXj, Xj, sizeof(float)*n, cudaMemcpyHostToDevice));  
    gpuErrchk(cudaMemcpy(dXij, Xij, sizeof(float)*m*n, cudaMemcpyHostToDevice)); 
    gpuErrchk(cudaMemcpy(dYi, Yi, sizeof(float)*m, cudaMemcpyHostToDevice)); 
    gpuErrchk(cudaMemcpy(dYj, Yj, sizeof(float)*n, cudaMemcpyHostToDevice));  

    dim3 threadsPerBlock(32,32);
    dim3 numBlocks((m+threadsPerBlock.x-1)/threadsPerBlock.x, (n+threadsPerBlock.y-1)/threadsPerBlock.y);
    rbf<<<numBlocks, threadsPerBlock>>>(m, n, dK, ldk, dXi, dXj, dXij, ldxij, gamma, dYi, dYj);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(K, dK, sizeof(float)*m*n, cudaMemcpyDeviceToHost));    
    cudaDeviceSynchronize();
}


void TC_GEMM(int rk, int n, int k, int d, float alpha, float beta, float *A, int lda, float *B, int ldb, float *C, int ldc){
    gpuErrchk(cudaMemcpy(dA, A, sizeof(float)*n*d, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dB, B, sizeof(float)*d*k, cudaMemcpyHostToDevice));

    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, d, &alpha, dA, CUDA_R_32F, lda,
            dB, CUDA_R_32F, ldb, &beta, dC, CUDA_R_32F, ldc,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
    
    gpuErrchk(cudaMemcpy(C, dC, sizeof(float)*n*k, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}


void cuda_finalize(){
    cudaFree(dXi);
    cudaFree(dXj);
    cudaFree(dXij);
    cudaFree(dYi);
    cudaFree(dYj);
    cudaFree(dK);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    // cudaFree(mem);
	// cudaFree(half_mem);
}






































// void Kernel(int rk, int m, int n, float *Xi, float *Xj, float *Yi, float *Yj, float *Xij, int ldxij, float *K, int ldk, float gamma){
//     float *dXi, *dXj, *dYi, *dYj, *dK, *dXij;
//     gpuErrchk(cudaMalloc(&dXi, sizeof(float)*m));
//     gpuErrchk(cudaMalloc(&dXj, sizeof(float)*n));
//     gpuErrchk(cudaMalloc(&dYi, sizeof(float)*m));
//     gpuErrchk(cudaMalloc(&dYj, sizeof(float)*n));
//     gpuErrchk(cudaMalloc(&dXij, sizeof(float)*m*n));
//     gpuErrchk(cudaMalloc(&dK, sizeof(float)*m*n));

//     gpuErrchk(cudaMemcpy(dXi, Xi, sizeof(float)*m, cudaMemcpyHostToDevice));  
//     gpuErrchk(cudaMemcpy(dXj, Xj, sizeof(float)*n, cudaMemcpyHostToDevice));  
//     gpuErrchk(cudaMemcpy(dYi, Yi, sizeof(float)*m, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(dYj, Yj, sizeof(float)*n, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(dXij, Xij, sizeof(float)*m*n, cudaMemcpyHostToDevice));

//     dim3 threadsPerBlock(32,32);
//     dim3 numBlocks((m+threadsPerBlock.x-1)/threadsPerBlock.x, (n+threadsPerBlock.y-1)/threadsPerBlock.y);
//     rbf<<<numBlocks, threadsPerBlock>>>(m, n, dK, ldk, dXi, dXj, dXij, ldxij, gamma, dYi, dYj);
    
//     // if(rk==0){  
//     //     printMatrixDeviceBlock("dK.csv", m, n, dK, ldk);
//     //     printMatrixDeviceBlock("dXij.csv", m, n, dXij, ldxij);
//     //     printVectorDeviceBlock("dXi.csv", m, dXi);
//     //     printVectorDeviceBlock("dXj.csv", n, dXj);
//     //     printVectorDeviceBlock("dYi.csv", m, dYi);
//     //     printVectorDeviceBlock("dYj.csv", n, dYj);
//     // } 

//     gpuErrchk(cudaMemcpy(K, dK, sizeof(float)*m*n, cudaMemcpyDeviceToHost));
//     cudaDeviceSynchronize();
//     cudaFree(dXi);
//     cudaFree(dXj);
//     cudaFree(dYi);
//     cudaFree(dYj);
//     cudaFree(dXij);
//     cudaFree(dK);
// }
