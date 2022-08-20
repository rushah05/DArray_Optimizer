// #pragma once
#include <cassert>
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

__global__
void  getR(int m, int n, float *da, int lda, float *dr, int ldr)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m&&j < n){
		if (i <= j){
			dr[i + j*ldr] = da[i + j*lda];
		}
	}
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
void rbf( int m, int n, FloatType *buf, int ldb,
				 FloatType *XI, FloatType *XJ, FloatType *XIJ, int ldxij,
				 GammaType gamma, FloatType *YI, FloatType *YJ)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	int j=blockIdx.y*blockDim.y + threadIdx.y;

	if (i<m && j<n) {
		buf[i+j*ldb] = YI[i]*YJ[j]*__expf(-gamma*(XI[i] + XJ[j] - 2*XIJ[i+j*ldxij]));
		// printf("[m,n]=[%d,%d], [i,j]=[%d,%d], buf[]=%.4f, XI[]=%.4f, XJ[]=%.4f, XIJ[]=%.4f\n", m, n, i, j, buf[i+j*ldb],
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

void transpose(float *dA, int lda, int m, int n)
{
	float *dC;
	gpuErrchk(cudaMalloc(&dC, sizeof(float)*m*n));
	float alpha = 1.0;
	float beta = 0.0;
	cublasSgeam(handle,
		CUBLAS_OP_T, CUBLAS_OP_T,
		m, n,
		&alpha,
		dA, lda,
		&beta,
		dA, lda,
		dC, m);
	cudaMemcpyAsync(dA, dC, m * n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
	cudaFree(dC);
}

__global__ void gemm(float *dA, int lda, float *dB, int ldb, float *dC, int ldc, int m, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < m && j < n) {
        float sum = 0.0f;
        for(int d=0; d<k; ++d){
            sum += dA[i+d*lda] * dB[d+j*ldb];
        }
        dC[i+j*ldc] = sum;
    }
}

void LRA(int rank, int lm, int ln, int ld, int lk, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* A, int lda, float gamma, float* O, int ldo){
    // printf("[Rank::%d] lm::%d, ln::%d, ld::%d, lk::%d \nldxi::%d, ldxj::%d, lda::%d, ldo::%d\n", rank, lm, ln, ld, lk, ldxi, ldxj, lda, ldo);
    const int B=8192;
    float *dXi, *dXj, *dXij, *dYi, *dYj, *dk, *dO, *dA;
    gpuErrchk(cudaMalloc(&dXi, sizeof(float)*lm*ld));
    gpuErrchk(cudaMalloc(&dXj, sizeof(float)*ln*ld));
    gpuErrchk(cudaMalloc(&dYi, sizeof(float)*lm));
    gpuErrchk(cudaMalloc(&dYj, sizeof(float)*ln));
    gpuErrchk(cudaMalloc(&dXij, sizeof(float)*B*B));
    gpuErrchk(cudaMalloc(&dk, sizeof(float)*B*B));
    gpuErrchk(cudaMalloc(&dO, sizeof(float)*ln*lk));
    gpuErrchk(cudaMalloc(&dA, sizeof(float)*lm*lk));
    gpuErrchk(cudaMemset(dA, 0.0, lm*lk));
    gpuErrchk(cudaMemcpy(dXi, Xi, sizeof(float)*lm*ld, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dXj, Xj, sizeof(float)*ln*ld, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dYi, Yi, sizeof(float)*lm, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dYj, Yj, sizeof(float)*ln, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dO, O, sizeof(float)*ln*lk, cudaMemcpyHostToDevice));
    
    float *dXi_sqr, *dXj_sqr;
    gpuErrchk(cudaMalloc(&dXi_sqr, sizeof(float)*B));
    gpuErrchk(cudaMalloc(&dXj_sqr, sizeof(float)*B));
    float sone=1.0f;
    float szero=0.0f;
    dim3 threadsPerBlock(32,32);
   

    for(int i=0; i<lm; i+=B){
        int ib=min(B, lm-i);
        for(int j=0; j<ln; j+=B){
            int jb=min(B, ln-j);
            vecnorm<<<(ib+63)/64, 64>>>(&dXi[i], ldxi, dXi_sqr, ib, ld);
	        vecnorm<<<(jb+63)/64, 64>>>(&dXj[j], ldxj, dXj_sqr, jb, ld);
            stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, ld, &sone, &dXi[i], CUDA_R_32F, ldxi,
                            &dXj[j], CUDA_R_32F, ldxj, &szero, dXij, CUDA_R_32F, ib,
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
            if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
            // printMatrixDeviceBlock("Xij.csv", ib, jb, dXij, B);
            // printVectorDeviceBlock("Xi_sqr.csv", ib, dXi_sqr);
            // printVectorDeviceBlock("Xj_sqr.csv", jb, dXj_sqr);
            // printVectorDeviceBlock("Yi.csv", ib, &dYi[i]);
            // printVectorDeviceBlock("Yj.csv", jb, &dYj[j]);

            dim3 numBlocks((ib+threadsPerBlock.x-1)/threadsPerBlock.x, (jb+threadsPerBlock.y-1)/threadsPerBlock.y );
            rbf<<<numBlocks, threadsPerBlock>>>(ib, jb, dk, ib, dXi_sqr, dXj_sqr, dXij, _IOFBF, gamma, &dYi[i], &dYj[j]);

            // printMatrixDeviceBlock("K.csv", ib, jb, dk, ib);
            // printMatrixDeviceBlock("O.csv", jb, lk, &dO[j], ldo);
             stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, lk, jb, &sone, dk, CUDA_R_32F, ib,
                            &dO[j], CUDA_R_32F, ldo, &szero, &dA[i], CUDA_R_32F, lda,
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
            if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
        }
    }

    // if(rank == 0) printMatrixDeviceBlock("A0.csv", lm, lk, dA, lda);
    gpuErrchk(cudaMemcpy(A, dA, sizeof(float)*lm*lk, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cudaFree(dXi);
    cudaFree(dXj);
    cudaFree(dXij);
    cudaFree(dk);
    cudaFree(dO);
    cudaFree(dA);
    cudaFree(dXi_sqr);
    cudaFree(dXj_sqr);
}


// void LRA(int rank, int lm, int ln, int ld, int lk, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* A, int lda, float gamma, float* O, int ldo){
//     // printf("[Rank::%d] lm::%d, ln::%d, ld::%d, lk::%d \nldxi::%d, ldxj::%d, lda::%d, ldo::%d\n", rank, lm, ln, ld, lk, ldxi, ldxj, lda, ldo);
//     const int B=8192;
//     float *dXi, *dXj, *dXij, *dYi, *dYj, *dk, *dO, *dA;
//     gpuErrchk(cudaMalloc(&dXi, sizeof(float)*lm*ld));
//     gpuErrchk(cudaMalloc(&dXj, sizeof(float)*ln*ld));
//     gpuErrchk(cudaMalloc(&dYi, sizeof(float)*lm));
//     gpuErrchk(cudaMalloc(&dYj, sizeof(float)*ln));
//     gpuErrchk(cudaMalloc(&dXij, sizeof(float)*B*B));
//     gpuErrchk(cudaMalloc(&dk, sizeof(float)*B*B));
//     gpuErrchk(cudaMalloc(&dO, sizeof(float)*ln*lk));
//     gpuErrchk(cudaMalloc(&dA, sizeof(float)*lm*lk));
//     gpuErrchk(cudaMemset(dA, 0.0, lm*lk));
//     gpuErrchk(cudaMemcpy(dXi, Xi, sizeof(float)*lm*ld, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(dXj, Xj, sizeof(float)*ln*ld, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(dYi, Yi, sizeof(float)*lm, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(dYj, Yj, sizeof(float)*ln, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(dO, O, sizeof(float)*ln*lk, cudaMemcpyHostToDevice));
    
//     float *dXi_sqr, *dXj_sqr;
//     gpuErrchk(cudaMalloc(&dXi_sqr, sizeof(float)*B));
//     gpuErrchk(cudaMalloc(&dXj_sqr, sizeof(float)*B));
//     float sone=1.0f;
//     float szero=0.0f;
//     dim3 threadsPerBlock(32,32);
   

//     for(int i=0; i<lm; i+=B){
//         int ib=min(B, lm-i);
//         for(int j=0; j<ln; j+=B){
//             int jb=min(B, ln-j);
//             vecnorm<<<(ib+63)/64, 64>>>(dXi, ldxi, dXi_sqr, ib, ld);
// 	        vecnorm<<<(jb+63)/64, 64>>>(dXj, ldxj, dXj_sqr, jb, ld);
//             stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, ld, &sone, &dXi[i], CUDA_R_32F, ldxi,
//                             &dXj[j], CUDA_R_32F, ldxj, &szero, dXij, CUDA_R_32F, ib,
//                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
//             if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
//             // printMatrixDeviceBlock("Xij.csv", ib, jb, dXij, B);
//             // printVectorDeviceBlock("Xi_sqr.csv", ib, dXi_sqr);
//             // printVectorDeviceBlock("Xj_sqr.csv", jb, dXj_sqr);
//             // printVectorDeviceBlock("Yi.csv", ib, &dYi[i]);
//             // printVectorDeviceBlock("Yj.csv", jb, &dYj[j]);

//             dim3 numBlocks((ib+threadsPerBlock.x-1)/threadsPerBlock.x, (jb+threadsPerBlock.y-1)/threadsPerBlock.y );
//             rbf<<<numBlocks, threadsPerBlock>>>(ib, jb, dk, ib, dXi_sqr, dXj_sqr, dXij, _IOFBF, gamma, &dYi[i], &dYj[j]);

//             // printMatrixDeviceBlock("K.csv", ib, jb, dk, ib);
//             // printMatrixDeviceBlock("O.csv", jb, lk, &dO[j], ldo);
//              stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, lk, jb, &sone, dk, CUDA_R_32F, ib,
//                             &dO[j], CUDA_R_32F, ldo, &szero, &dA[i], CUDA_R_32F, lda,
//                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
//             if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
//         }
//     }

//     // if(rank == 0) printMatrixDeviceBlock("A0.csv", lm, lk, dA, lda);
//     gpuErrchk(cudaMemcpy(A, dA, sizeof(float)*lm*lk, cudaMemcpyDeviceToHost));
//     cudaDeviceSynchronize();
//     cudaFree(dXi);
//     cudaFree(dXj);
//     cudaFree(dXij);
//     cudaFree(dk);
//     cudaFree(dO);
//     cudaFree(dA);
//     cudaFree(dXi_sqr);
//     cudaFree(dXj_sqr);
// }


void SGEQRF(int m, int n, float *Q, int ldq){
    float *dQ, *d_tau, *d_work;
    int *d_info, info;
    int lwork_geqrf = 0;
	int lwork_orgqr = 0;
    int lwork = 0;

    gpuErrchk(cudaMalloc(&dQ, sizeof(float)*m*n));
    gpuErrchk(cudaMalloc(&d_tau, sizeof(float)*n));
	gpuErrchk(cudaMalloc((void**)&d_info, sizeof(int)));
    gpuErrchk(cudaMemcpy(dQ, Q, sizeof(float)*m*n, cudaMemcpyHostToDevice));

    statusH = cusolverDnSgeqrf_bufferSize(csHandle, m, n, dQ, ldq, &lwork_geqrf);
    assert(statusH == CUSOLVER_STATUS_SUCCESS);
    statusH = cusolverDnSorgqr_bufferSize(csHandle, m, n, n, dQ, ldq, d_tau, &lwork_orgqr);
    assert(statusH == CUSOLVER_STATUS_SUCCESS);
    // printf("lwork_geqrf::%d, lwork_orgqr::%d\n", lwork_geqrf, lwork_orgqr);

	lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
	cudaMalloc(&d_work, sizeof(int)*lwork);

    statusH = cusolverDnSgeqrf(csHandle, m, n, dQ, ldq, d_tau, d_work, lwork, d_info);
    assert(statusH == CUSOLVER_STATUS_SUCCESS);
    statusH = cusolverDnSorgqr(csHandle, m, n, n, dQ, ldq, d_tau, d_work, lwork, d_info);
    assert(statusH == CUSOLVER_STATUS_SUCCESS);
    gpuErrchk(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info!=0) {
        printf("cusolver Sorgqr fail; info=%d\n", info);
        exit(1); 
    }
       
    gpuErrchk(cudaMemcpy(Q, dQ, sizeof(float)*m*n, cudaMemcpyDeviceToHost));

    cudaFree(dQ);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(d_info);
}

void SORMQR(int m, int n, float *Q, int ldq, float *RQ, int ldrq){
    float *dQ, *dRQ, *d_tau, *d_work;
    int *d_info, info;
	int lwork_ormqr=0;

    gpuErrchk(cudaMalloc(&dQ, sizeof(float)*m*n));
    gpuErrchk(cudaMalloc(&dRQ, sizeof(float)*m*n));
    gpuErrchk(cudaMalloc(&d_tau, sizeof(float)*m));
	gpuErrchk(cudaMalloc((void**)&d_info, sizeof(int)));
    gpuErrchk(cudaMemcpy(dQ, Q, sizeof(float)*m*n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRQ, RQ, sizeof(float)*n*n, cudaMemcpyHostToDevice));

    statusH = cusolverDnSormqr_bufferSize(csHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n, n, dQ, ldq, d_tau, dRQ, ldrq, &lwork_ormqr);
	assert(statusH == CUSOLVER_STATUS_SUCCESS);
    cudaMalloc(&d_work, sizeof(int)*lwork_ormqr);

    cusolverDnSormqr(csHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n, n, dQ, ldq, d_tau, dRQ, ldrq, d_work, lwork_ormqr, d_info);
    gpuErrchk(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info!=0) {
        printf("cusolver Sormqr fail; info=%d\n", info);
        exit(1); 
    }

    gpuErrchk(cudaMemcpy(RQ, dRQ, sizeof(float)*m*n, cudaMemcpyDeviceToHost));
    cudaFree(dQ);
    cudaFree(dRQ);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(d_info);
}


void TCGemm(int rank, int m, int n, int k, float* A, int lda, float* B, int ldb, float alpha, float beta, float* C, int ldc){
    // printf("[Rank::%d] m::%d, n::%d, k::%d lda::%d, ldb::%d, ldc::%d\n", rank, m, n, k, lda, ldb, ldc);
    float *da, *db, *dc;
    gpuErrchk(cudaMalloc(&da, sizeof(float)*m*k));
    gpuErrchk(cudaMalloc(&db, sizeof(float)*k*n));
    gpuErrchk(cudaMalloc(&dc, sizeof(float)*m*n));
    gpuErrchk(cudaMemcpy(da, A, sizeof(float)*m*k, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, B, sizeof(float)*k*n, cudaMemcpyHostToDevice));

    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, da, CUDA_R_32F, lda,
                        db, CUDA_R_32F, ldb, &beta, dc, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
    if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
    gpuErrchk(cudaMemcpy(C, dc, sizeof(float)*m*n, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc); 
}

void Chol(int k, float *C, int ldc){
    float *Cd, *d_work;
    gpuErrchk(cudaMalloc(&Cd, sizeof(float)*k*k));
    gpuErrchk(cudaMemcpy(Cd, C, sizeof(float)*k*k, cudaMemcpyHostToDevice));

    int info, *d_info;
    int lwork=0; 
    gpuErrchk(cudaMalloc(&d_info, sizeof(int)));
    statusH = cusolverDnSpotrf_bufferSize(csHandle, CUBLAS_FILL_MODE_LOWER, k, Cd, ldc, &lwork);
    assert(statusH == CUSOLVER_STATUS_SUCCESS);

    gpuErrchk(cudaMalloc(&d_work, sizeof(float)*lwork));
    statusH = cusolverDnSpotrf(csHandle, CUBLAS_FILL_MODE_LOWER, k, Cd, ldc, d_work, lwork, d_info);
    assert(statusH == CUSOLVER_STATUS_SUCCESS);
    cudaFree(d_work);
    gpuErrchk(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) );
    if (info!=0) {
        printf("Cholesky fail; info=%d\n", info);
        // exit(1); 
    }
    cudaFree(d_info); 
}













