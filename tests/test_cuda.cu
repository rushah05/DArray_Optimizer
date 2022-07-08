#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

/*
cd ../tests
module load GCC/7.3.0
nvcc -c test_cuda.cu -o cuda.o
cd ../build
make
module load GCC
./test_lra
*/

float *mem;
__half *half_mem;
float *dA, *dB, *dC;
__half *halfA, *halfB;
cublasHandle_t handle;

template<typename T>
void printMatrixDeviceBlock(char *filename,int m, int n, T* dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    //printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
	//cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    //printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

void init()
{
	cudaMalloc(&mem, sizeof(float)*65536*65536);
	cudaMalloc(&half_mem, sizeof(__half)*65536*65536);
	cublasCreate(&handle);
}

void finalize()
{
	cudaFree(mem);
	cudaFree(half_mem);
}

void host2device(int m, int n, int k, float *hA, float *hB)
{
	dA = mem;
	dB = mem+sizeof(float)*m*k;
	dC = mem+sizeof(float)*k*n+sizeof(float)*m*k;
	cudaMemcpy(dA, hA, sizeof(float)*m*k, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(float)*k*n, cudaMemcpyHostToDevice);
	halfA = half_mem;
	halfB = half_mem+sizeof(__half)*m*k;
}

void device2host(int m, int n, float *hC)
{
	cudaMemcpy(hC, dC, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
}


__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

void transpose(float *dA, int lda, int m, int n)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
	float *dC;
	cudaMalloc(&dC, sizeof(float)*m*n);
	float alpha = 1.0;
	float beta = 0.0;
	cublasSgeam(handle,
		CUBLAS_OP_T, CUBLAS_OP_T,
		m, n,
		&alpha,
		dA, lda,
		&beta,
		dA, lda,
		dC, lda);
	cudaMemcpy(dA, dC, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(dC);
}

void TCGEMM(int m, int n, int k, float alpha, float beta, float *hA, float *hB, float *hC)
{
	printf("m,n,k=%d,%d,%d\n", m, n, k);
	host2device(m,n,k,hA,hB);
	//printMatrixDeviceBlock("dA.csv",m, k, dA, m);
	//printMatrixDeviceBlock("dB.csv",k, n, dB, n);
	dim3 gridDimA((m+31)/32, (k+31)/31 );
	dim3 blockDim(32,32);
	s2h<<<gridDimA, blockDim>>>(m,k, dA, m, halfA, m);
	dim3 gridDimB((k+31)/32, (n+31)/31);
	s2h<<<gridDimB, blockDim>>>(k,n, dB, k, halfB, k);

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        &alpha, halfA, CUDA_R_16F, m, halfB, CUDA_R_16F, k,
        &beta, dC, CUDA_R_32F, m, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

	//transpose(dC, m, m, n);

	device2host(m,n,hC);
}

void test_cuda()
{
	printf("Yes\n");
}


// // alpha=1.0, beta=0.0
// __global__
// void rbf(float gamma, float *Xi, float *Xj, int ldx, float &alpha, float &beta, float *Yi, float *Yj, int d, int n, float *K, int ldk){
// 	float *Xi_sqr=(float*)malloc(sizeof(float)*n*n);
// 	cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &alpha, Xi, CUDA_R_32F, n, Xi, CUDA_R_32F, d,
//         &beta, Xi_sqr, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

// 	float *Xj_sqr=(float*)malloc(sizeof(float)*n*n);
// 	cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &alpha, Xj, CUDA_R_32F, n, Xj, CUDA_R_32F, d,
//         &beta, Xj_sqr, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

// 	float *Xij=(float*)malloc(sizeof(float)*n*n);
// 	alpha=-2.0;
// 	cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &alpha, Xi, CUDA_R_32F, n, Xj, CUDA_R_32F, d,
//         &beta, Xij, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// }
