// #pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>

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
			else fprintf(f, " ,");
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


template<typename FloatType, typename GammaType>
__global__
void rbf( int m, int n, FloatType *buf, int ldb, FloatType *Xi, FloatType *Xj, FloatType *Xij, int ldxij, GammaType gamma, FloatType *Yi, FloatType *Yj)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	int j=blockIdx.y*blockDim.y + threadIdx.y;

	if (i<m && j<n) {
		buf[i+j*ldb] = Yi[i]*Yj[j]*__expf(-gamma*(Xi[i] + Xj[j] - 2*Xij[i+j*ldxij]));
		// printf("[i,j]=[%d,%d], buf[]=%.4f, XI[]=%.4f, XJ[]=%.4f, XIJ[]=%.4f\n", i, j, buf[i+j*ldb],
		// 	   XI[i], XJ[j], XIJ[i+j*ldxij]);
	}
}


template<typename FloatType>
__global__ 
void vecnorm(FloatType *Zd, int ldz, FloatType *ZI, int m, int k){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if( i<m ) {
		FloatType sum = 0.0f;

        #pragma unroll (4)
		for( int j=0; j<k; j++ )
			sum += Zd[i+j*ldz]*Zd[i+j*ldz];

		ZI[i] = sum;
	}
}

__global__ 
void fnorm( int m, int n, double *buf, int B, double *XI, double *XJ, double *XIJ, int ldxij, double gamma, double *YI, double *YJ, double *acc){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	int j=blockIdx.y*blockDim.y + threadIdx.y;

	if (i<m && j<n) 
		// buf[i+j*ldb] = YI[i]*YJ[j]*__expf(-gamma*(XI[i] + XJ[j] - 2*XIJ[i+j*ldxij]));
		acc += 0;
}


__global__ void s2h(float *Z, __half *hZ, int ldz, int m, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;

    if(i < m && j < n){
        hZ[i+j*ldz] = __float2half(Z[i+j*ldz]);
    }
}

__global__ void h2s(__half *hZ, float *Z, int ldz, int m, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;

    if(i < m && j < n){
        Z[i+j*ldz] = __half2float(hZ[i+j*ldz]);
    }
}


__global__ void s2h_vec(float *Z, __half *hZ, int m){
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i < m){
        hZ[i] = __float2half(Z[i]);
    }
}




void lra(int rank, int gn, int ln, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KO, int ldk, float gamma, float* Omega, int ldo){
    const int B=16384;
    if(rank ==0) printf("gn::%d, ln::%d, d::%d, k::%d\nldxi::%d, ldxj::%d, ldk::%d, ldo::%d\n", gn, ln, d, k, ldxi, ldxj, ldk, ldo);
    float *dXi, *dXj, *dYi, *dYj, *dO, *dKO;
    gpuErrchk(cudaMalloc(&dXi, sizeof(float)*ln*d));
    gpuErrchk(cudaMalloc(&dXj, sizeof(float)*gn*d));
    gpuErrchk(cudaMalloc(&dYi, sizeof(float)*ln));
    gpuErrchk(cudaMalloc(&dYj, sizeof(float)*gn));
    gpuErrchk(cudaMalloc(&dO, sizeof(float)*gn*k));
    gpuErrchk(cudaMalloc(&dKO, sizeof(float)*ln*k));
    gpuErrchk(cudaMemcpy(dXi, Xi, sizeof(float)*ln*d, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dXj, Xj, sizeof(float)*gn*d, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dYi, Yi, sizeof(float)*ln, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dYj, Yj, sizeof(float)*gn, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dO, Omega, sizeof(float)*gn*k, cudaMemcpyHostToDevice));

    float *dXi_sqr, *dXj_sqr, *dk, *dXij;
    gpuErrchk(cudaMalloc(&dXi_sqr, sizeof(float)*B));
    gpuErrchk(cudaMalloc(&dXj_sqr, sizeof(float)*B));
    gpuErrchk(cudaMalloc(&dk, sizeof(float)*B*B));
    gpuErrchk(cudaMalloc(&dXij, sizeof(float)*B*B));
    cudaDeviceSynchronize();

    float sone=1.0f;
    float szero=0.0f;
    for (int i=0; i<ln; i+=B) {
		int ib = min(B, ln-i);
		for (int j=0; j<gn; j+=B) {
			int jb = min(B, gn-j);

			// printf("[i=%d j=%d] ib::%d, jb::%d\n", i,j, ib, jb);
			// step 1: populate XI, XJ, XIJ
			vecnorm<<<(B+63)/64, 64>>>(&dXi[i], ldxi, dXi_sqr, ib, d);
            gpuErrchk( cudaPeekAtLastError() );
			vecnorm<<<(B+63)/64, 64>>>(&dXj[j], ldxj, dXj_sqr, jb, d);
			gpuErrchk( cudaPeekAtLastError() );

			// XIJ is column major!!
			// printf("ib=%d jb=%d d=%d ldz=%d\n", ib, jb, d, ldz);
			stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, d,
								   &sone, &dXi[i], CUDA_R_32F,ldxi,
								   &dXj[j], CUDA_R_32F,ldxj, &szero,
								   dXij, CUDA_R_32F, ib,
								   CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
			if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
			gpuErrchk( cudaPeekAtLastError() );

			dim3 threadsPerBlock(32,32);
			dim3 numBlocks((ib+threadsPerBlock.x-1)/threadsPerBlock.x,
							(jb+threadsPerBlock.y-1)/threadsPerBlock.y );

			// printf("ib=%d, jb=%d, B=%d, TPB.(x,y)=(%d,%d), B.(x,y)=(%d,%d)\n",
			// 	   ib, jb, B, threadsPerBlock.x, threadsPerBlock.y,
			// 	   numBlocks.x, numBlocks.y);
			rbf<<<numBlocks, threadsPerBlock>>>( ib, jb, dk, ib, dXi_sqr, dXj_sqr, dXij, ib, gamma, &dYi[i], &dYj[j]);
			gpuErrchk( cudaPeekAtLastError());
			gpuErrchk( cudaDeviceSynchronize());

            stat = cublasGemmEx( handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, k, jb, 
                                &sone, dk, CUDA_R_32F, B, &dO[j], CUDA_R_32F, ldo,
                                &sone, &dKO[i], CUDA_R_32F, ldk, 
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
        }
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(KO, dKO, sizeof(float)*ln*k, cudaMemcpyDeviceToHost));
    cudaPeekAtLastError();
}



void SGEQRF(int *m, int *n, float *Q, int *ldq, float *tau, float *work, int *lwork, int *info){

}
