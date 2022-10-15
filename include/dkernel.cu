// #pragma once
#include <cassert>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <string>
#define B 8192

cublasHandle_t handle;
cusolverDnHandle_t csHandle;
cudaStream_t stream;
cudaError_t cudaStat;
cublasStatus_t stat;
cusolverStatus_t statusH = CUSOLVER_STATUS_SUCCESS;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
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

template<typename T>
void copyMatrixDeviceToHost(int m, int n, T* dsrc, int ldsrc, T* ddest, int lddest){
    T *hsrc = (T*)malloc(sizeof(T)*m*n);
    cudaMemcpy(hsrc, dsrc, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    for(int i = 0;i<m;i++){
        for(int j = 0;j<n;j++){
            ddest[i+j*lddest] = hsrc[i+j*ldsrc];
        }
    }
}


__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

template<typename FloatType>
__global__ void
vecnorm(FloatType *Zd, int ldz, FloatType *ZI, int m, int k)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if( i<m ) {
		FloatType sum = 0;
        #pragma unroll (4)
		for( int j=0; j<k; j++ )
			sum += Zd[i+j*ldz]*Zd[i+j*ldz];
		ZI[i] = sum;
	}
}

template<typename FloatType, typename GammaType>
__global__
void rbf( int m, int n, FloatType *buf, int ldb, FloatType *XI, FloatType *XJ, FloatType *XIJ, int ldxij,
				 GammaType gamma, FloatType *YI, FloatType *YJ)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	int j=blockIdx.y*blockDim.y + threadIdx.y;

	if (i<m && j<n) {
		buf[i+j*ldb] = YI[i]*YJ[j]*__expf(-gamma*(XI[i] + XJ[j] - 2*XIJ[i+j*ldxij]));
        // printf("[i,j]=[%d,%d], buf[]=%.4f, XI[]=%.4f, XJ[]=%.4f, XIJ[]=%.4f\n", i, j, buf[i+j*ldb],
		// 	   XI[i], XJ[j], XIJ[i+j*ldxij]);
    }
}


void rbf_kernel(int rank, int m, int n, int d, float *Xi, int ldxi, float *Xj, int ldxj, float *Yi, float *Yj, float *K, int ldk, float gamma){
    // if(rank == 0) printf("m=%d, n=%d, d=%d, ldxi=%d, ldxj=%d, ldk=%d \n", m, n, d, ldxi, ldxj, ldk);
    float *dXi, *dXj, *dYi, *dYj;
    gpuErrchk(cudaMallocManaged( &dXi, sizeof(float)*m*d));
    gpuErrchk(cudaMallocManaged( &dXj, sizeof(float)*n*d));
    gpuErrchk(cudaMallocManaged( &dYi, sizeof(float)*m));
    gpuErrchk(cudaMallocManaged( &dYj, sizeof(float)*n));
    gpuErrchk(cudaMemcpy(dXi, Xi, sizeof(float)*m*d, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dYi, Yi, sizeof(float)*m, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dXj, Xj, sizeof(float)*n*d, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dYj, Yj, sizeof(float)*n, cudaMemcpyHostToDevice) );
    float *dK, *dXij, *dXi_sqr, *dXj_sqr;
    gpuErrchk(cudaMallocManaged( &dK, sizeof(float)*B*B ));
    gpuErrchk(cudaMallocManaged( &dXij, sizeof(float)*B*B ));
    gpuErrchk(cudaMallocManaged( &dXi_sqr, sizeof(float)*B ));
    gpuErrchk(cudaMallocManaged( &dXj_sqr, sizeof(float)*B ));
    gpuErrchk( cudaDeviceSynchronize() );

    float one=1, zero=0;
    for(int i=0; i<m; i+=B){
        int ib=std::min(B, m-i);
        vecnorm<<<(B+63)/64, 64>>>(&dXi[i], m, dXi_sqr, ib, d);

        for(int j=0; j<n; j+=B){
            int jb=std::min(B, n-j);
            vecnorm<<<(B+63)/64, 64>>>(&dXj[j], n, dXj_sqr, jb, d);
            gpuErrchk( cudaPeekAtLastError() );

            if(d>=256){
                stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, d, &one, &dXi[i], CUDA_R_32F, m,
                    &dXj[j], CUDA_R_32F, n, &zero, dXij, CUDA_R_32F, ib,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
                if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
            }else{
                stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, d, &one, &dXi[i], m, &dXj[j], n, &zero, dXij, ib);		
                if (stat != CUBLAS_STATUS_SUCCESS) 
                printf ("cublasSgemm failed %s\n", __LINE__);
            }

            dim3 threadsPerBlock(32,32);
            dim3 numBlocks( (ib+threadsPerBlock.x-1)/threadsPerBlock.x,
                            (jb+threadsPerBlock.y-1)/threadsPerBlock.y );
            rbf<<<numBlocks, threadsPerBlock>>>(ib, jb, dK, ib, dXi_sqr, dXj_sqr, dXij, ib, gamma, &dYi[i], &dYj[j]);
            gpuErrchk( cudaPeekAtLastError() );

            copyMatrixDeviceToHost(ib, jb, dK, ib, &K[i+j*ldk], ldk);
        }
    }   

    cudaFree(dXi);
    cudaFree(dXj);
    cudaFree(dYi);
    cudaFree(dYj);
    cudaFree(dK);
    cudaFree(dXij);
    cudaFree(dXi_sqr);
    cudaFree(dXj_sqr);
    gpuErrchk( cudaDeviceSynchronize() );
}



void tc_gemm(int rank, int m, int k, int n, float *A, int lda, float *BB, int ldb, float *C, int ldc, float alpha=1, float beta=0){
    float *dA, *dB, *dC;
    gpuErrchk(cudaMallocManaged( &dA, sizeof(float)*m*n));
    gpuErrchk(cudaMallocManaged( &dB, sizeof(float)*n*k));
    gpuErrchk(cudaMallocManaged( &dC, sizeof(float)*m*k));
    gpuErrchk(cudaMemcpy(dA, A, sizeof(float)*m*n, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dB, BB, sizeof(float)*n*k, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemset(dC, 0.0, sizeof(float)*m*k));
    gpuErrchk(cudaDeviceSynchronize());

	for (int i=0; i<m; i+=B) {
		int ib = std::min(B, m-i);
		for (int j=0; j<n; j+=B) {
			int jb = std::min(B, n-j);

            if(jb>=256){
                stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, k, jb, &alpha, &dA[i+j*lda], CUDA_R_32F, lda,
                    &dB[j], CUDA_R_32F, ldb, &beta, &dC[i], CUDA_R_32F, m,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
                if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
                
            }else{
                stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, k, jb, &alpha, &dA[i+j*lda], lda, &dB[j], ldb, 
                &beta, &dC[i], m);	

                if (stat != CUBLAS_STATUS_SUCCESS) 
                printf ("cublasSgemm failed %s\n", __LINE__);
            }
        }
    }

    copyMatrixDeviceToHost(m, k, dC, m, C, ldc);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaDeviceSynchronize();
}


void tc_sgemm(int rank, char transA, char transB, int m, int k, int n, float *A, int lda, float *BB, int ldb, float *C, int ldc, float alpha=1, float beta=0){
    float *dA, *dB, *dC;
    gpuErrchk(cudaMallocManaged( &dA, sizeof(float)*m*n));
    gpuErrchk(cudaMallocManaged( &dB, sizeof(float)*n*k));
    gpuErrchk(cudaMallocManaged( &dC, sizeof(float)*m*k));
    gpuErrchk(cudaMemcpy(dA, A, sizeof(float)*m*n, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dB, BB, sizeof(float)*n*k, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemset(dC, 0.0, sizeof(float)*m*k));
    gpuErrchk(cudaDeviceSynchronize());


    if(n>=256){
        if(transA=='N' && transB=='N'){
            stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, dA, CUDA_R_32F, lda,
            dB, CUDA_R_32F, ldb, &beta, dC, CUDA_R_32F, m,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
            if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx (N,N) failed %s\n", __LINE__);
        }else if(transA=='T' && transB=='N'){
            stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k, n, &alpha, dA, CUDA_R_32F, lda,
            dB, CUDA_R_32F, ldb, &beta, dC, CUDA_R_32F, m,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
            if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx (T,N) failed %s\n", __LINE__);
        }
    }else{
        if(transA=='N' && transB=='N'){
            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, dA, lda, dB, ldb, 
            &beta, dC, m);	
            if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasSgemm failed %s\n", __LINE__);
        }else if(transA=='T' && transB=='N'){
            stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k, n, &alpha, dA, lda, dB, ldb, 
            &beta, dC, m);	
            if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasSgemm failed %s\n", __LINE__);
        }
    }

    copyMatrixDeviceToHost(m, k, dC, m, C, ldc);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaDeviceSynchronize();
}
































