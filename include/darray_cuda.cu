// #pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

cublasHandle_t handle;
cublasStatus_t stat;
cudaStream_t stream;
	
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
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
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
        fprintf(f, "%d", i);
        fprintf(f, "::");
        fprintf(f, "%lf", ha[0]);
        fprintf(f, "\n");
    }

    fclose(f);
    free(ha);
}


// CUDA kernel to generate the matrix, block by block.
// the result will be store in buf, column major, m*n matrix,
// with LDA m.
// XIJ/buf are column major.
// could be improved by assigning more work to each thread.
template<typename FloatType, typename GammaType>
__global__
void rbf( int m, int n, FloatType *buf, int ldb, FloatType *Xi, FloatType *Xj, FloatType *Xij, 
int ldxij, GammaType gamma, FloatType *Yi, FloatType *Yj)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	int j=blockIdx.y*blockDim.y + threadIdx.y;

	if (i<m && j<n) {
		buf[i+j*ldb] = Yi[i]*Yj[j]*__expf(-gamma*(Xi[i] + Xj[j] - 2*Xij[i+j*ldxij]));
	}
}

template<typename FloatType>
__global__ void vecnorm(FloatType *Z, int ldz, FloatType *Zi, int m, int k){
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

__global__ void vecnormT(float *Z, int ldz, float *Zi, int k, int m){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < m){
        float sum=0.0f;
        #pragma unroll (4)
        for( int j=0; j<k; j++){
            sum+=Z[j+i*ldz]*Z[j+i*ldz];
        }
        Zi[i]=sum;
    }
}


void lra(int rank, int n, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KOmega, int ldk, float gamma, float* Omega, int ldo){
    const int B = 8192;
    float *dk, *dXij, *dXi, *dYi, *dXj, *dYj, *dXi_sqr, *dXj_sqr, *dO, *dKO;
    gpuErrchk(cudaMalloc(&dk, sizeof(float)*B*B));
    gpuErrchk(cudaMalloc(&dXij, sizeof(float)*B*B));
    gpuErrchk(cudaMalloc(&dO, sizeof(float)*B*k));
    gpuErrchk(cudaMalloc(&dKO, sizeof(float)*B*k));
    gpuErrchk(cudaMalloc(&dXi_sqr, sizeof(float)*B));
    gpuErrchk(cudaMalloc(&dXj_sqr, sizeof(float)*B));
    gpuErrchk(cudaMalloc(&dXi, sizeof(float)*B*d));
    gpuErrchk(cudaMalloc(&dXi, sizeof(float)*B*d));
    gpuErrchk(cudaMalloc(&dXj, sizeof(float)*d*B));
    gpuErrchk(cudaMalloc(&dYi, sizeof(float)*B));
    gpuErrchk(cudaMalloc(&dYj, sizeof(float)*B));
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    // printf("rank::%d, n::%d, d::%d, k::%d\n", rank, n, d, k);
    // char r=rank;
    // char name[] = { 'X','i','_',r,'.','t','x','t','\0' }; 
    // printMatrixDeviceBlock(name, n, d, dXi, ldxi);
    // name[1]='j';
    // printMatrixDeviceBlock(name, d, n, dXj, ldxj);

    float sone=1.0f;
	float szero=0.0f;
    for(int i=0; i<n; i+=B){
        int ib=min(B, n-i);
        for(int j=0; j<n; j+=B){
            int jb=min(B, n-i);
            gpuErrchk(cudaMemcpy(dXi, &Xi[i], sizeof(float)*ib*d, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(dXj, &Xj[j], sizeof(float)*d*jb, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(dYi, &Yi[i], sizeof(float)*ib, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(dYj, &Yj[j], sizeof(float)*jb, cudaMemcpyHostToDevice));
            cudaPeekAtLastError();
            // if(rank == 0) printf("rank::%d, (B+63)/64::%d, ldxi::%d, ib::%d, d::%d\n", rank, (B+63)/64, ldxi, ib, d);
            vecnorm<<<(B+63)/64, 64>>>(dXi, ib, dXi_sqr, ib, d);
            vecnormT<<<(B+63)/64, 64>>>(dXj, d, dXj_sqr, d, jb);
            // // char name[] = { 'X','i','_',r,'.','t','x','t','\0' }; 
            // // printVectorDeviceBlock(name, ib, dXi_sqr);
            // // name[1]='j';
            // // printVectorDeviceBlock(name, jb, dXj_sqr);
            if(d >= 128){
                stat=cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, jb, d, &sone, dXi, CUDA_R_32F, ib, dXj, CUDA_R_32F, d, &szero, dXij,
                    CUDA_R_32F, ib, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }else{    
                stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, jb, d, &sone, dXi, ib, dXj, d, &szero, dXij, ib);
            }
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("cublasSgemm failed %s\n", __LINE__);
            }
            // char name[] = { 'X','i','j','_', r,'.','t','x','t','\0' }; 
            // printMatrixDeviceBlock(name, ib, jb, dXij, ib);

            dim3 threadsPerBlock(32,32);
			dim3 numBlocks( (ib+threadsPerBlock.x-1)/threadsPerBlock.x,
							(jb+threadsPerBlock.y-1)/threadsPerBlock.y );

			rbf<<<numBlocks, threadsPerBlock>>>( ib, jb, dk, B, dXi, dXj, dXij, ib, gamma, &dYi[i], &dYj[j]);
            gpuErrchk(cudaPeekAtLastError());
            cudaDeviceSynchronize();

            gpuErrchk(cudaMemcpy(dO, &Omega[j], sizeof(float)*jb*k, cudaMemcpyHostToDevice));
            if (k>1) {
				if (k < 256) {
					cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, k, jb,
								&sone, dk, ib, dO, jb,
								&sone, dKO, ib);
				}
				else {
					cublasGemmEx( handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, k, jb, 
								  &sone, dk, CUDA_R_32F, ib, dO, CUDA_R_32F, jb,
					 			  &sone, dKO, CUDA_R_32F, ib, 
					 			  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
				}
			} else if (k==1) {
				cublasSgemv(handle, CUBLAS_OP_N, ib, jb, &sone, dk, ib, dO, 1, &sone, dKO, 1);
			}

			gpuErrchk( cudaPeekAtLastError() );
			cudaDeviceSynchronize();

            // gpuErrchk(cudaMemcpy(&KOmega[j], dKO, sizeof(float)*ib*d, cudaMemcpyDeviceToHost));

        }
    }

    cudaFree(dk);
    cudaFree(dXij);
	cudaFree(dXi);
    cudaFree(dXj);
    cudaFree(dYi);
    cudaFree(dYj);
    cudaFree(dXi_sqr);
    cudaFree(dXj_sqr);
}

