#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

cublasHandle_t handle;
cublasStatus_t stat;
float *mem;
__half *half_mem;

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
    // cudaMalloc(&mem, sizeof(float)*65536*65536);
	// cudaMalloc(&half_mem, sizeof(__half)*65536*65536);
}

void cuda_finalize(){
    // cudaFree(mem);
	// cudaFree(half_mem);
}

template<typename T>
void printMatrixDeviceBlock(char *filename,int m, int n, T* dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++){
        for(int j = 0;j<n;j++){
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
    free(ha);
}


template<typename T>
void printVectorDeviceBlock(char *filename, int m, T* dA)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i=0; i<m; i++){
        cudaMemcpy(&ha[0], &dA[i], sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(f, "%lf", ha[0]);
        fprintf(f, ",");
    }

    free(ha);
    fclose(f);
}


__global__ void s2h(){

}

__global__ void vecnorm(float *Z, int ldz, float *Zi, int m, int k){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < m){
        float sum=0.0f;
        #pragma unroll (4)
        for( int j=0; j<k; j++){
            sum+=Z[i+j*ldz]*Z[i+j*ldz];
        }
        Zi[i]=sum;
    }
}


void lra(int m, int n, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KOmega, int ldk, float* Omega, int ldo){
    const int B = 8192;
    float *dk, *dXij, *dXi, *dXj, *dY;
    gpuErrchk(cudaMalloc(&dk, sizeof(float)*B*B));
    gpuErrchk(cudaMalloc(&dXij, sizeof(float)*B*B));
    gpuErrchk(cudaMalloc(&dXi, sizeof(float)*B));
    gpuErrchk(cudaMalloc(&dXj, sizeof(float)*B));
    cudaDeviceSynchronize();
    // printf("&dXi[i]::%f, ldxi::%d, &dXj[j]::%f, ldxj::%d\n", dXi[0], ldxi, dXj[0], ldxj);
    // printf("Xi[6]::%f, Xj[6]::%f\n", Xi[6], Xj[6]);
	float sone=1.0f;
	float szero=0.0f;
    for(int i=0; i<m; i+=B){
        int ib=min(B, m-i);
        // printf("ib::%d, m::%d, i::%d, B::%d\n", ib, m, i, B);
        for(int j=0; j<n; j+=B){
            int jb=min(B, n-j);
            vecnorm<<<(B+63)/64, 64>>>(&Xi[i], ldxi, dXi, ib, d);
            vecnorm<<<(B+63)/64, 64>>>(&Xi[j], ldxi, dXj, jb, d);
            gpuErrchk(cudaPeekAtLastError());
            printf("ib::%d, jb::%d, ldxi::%d, ldxj::%d, &dXi[6]::%f, &dXj[6]::%f, Xij[6]::%f\n", ib, jb, ldxi, ldxj, &dXi[6], &dXj[6], &dXij[6]);
            // printf("&dXi[i]::%f, ldxi::%d, &dXj[j]::%f, ldxj::%d\n", &dXi[i], ldxi, &dXj[j], ldxj);

            // if(d >= 128){
            //     stat=cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, jb, d, &sone,
            //         &dXi[i], CUDA_R_32F, ib, &dXj[j], CUDA_R_32F, d, &szero, dXij,
            //         CUDA_R_32F, ib, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            // }else{    
            //      stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, jb, d, &sone, &dXi[i], 
            //                 ib, &dXj[j], d, &szero, dXij, ib);
            // }
            // if (stat != CUBLAS_STATUS_SUCCESS) {
            //     printf ("cublasSgemm failed %s\n", __LINE__);
            // }
            // gpuErrchk(cudaPeekAtLastError());
            //  printf ("cublasSgemm failed %s %d\n", __LINE__, stat);
        }
    }
    cudaFree(dk);
	cudaFree(dXi);
	cudaFree(dXj);
	cudaFree(dXij);
	cudaDeviceSynchronize();
}

