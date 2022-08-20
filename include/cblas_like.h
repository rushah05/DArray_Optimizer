#pragma once
#include "darray.h"

extern "C" {
    //BLAS
void sgemm_(const char *ta, const char *tb, int *m, int *n, int *k, float *alpha, float *A, int *lda,
            float *B, int *ldb, float *beta, float *C, int *ldc);
void dgemm_(const char *ta, const char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda,
            double *B, int *ldb, double *beta, double *C, int *ldc);

void strsm_(const char *side, const char *uplo, const char *transa, const char *diag, int *m, int *n, float *alpha, float *A, int *lda, float *B, int *ldb);

void ssyrk_(const char *uplo, const char* trans, int *n, int *k, float *alpha, float *A, int *lda, float *beta, float *c, int *ldc);

    // LAPACK
void spotrf_(const char *uplo, int *N, float *A, int *lda, int *info);
void sgetrf_(int *M, int *N, float *A, int *lda, int *ipiv, int *info);
void sgeqrf_(int *M, int *N, float *A, int *lda, float *tau, float *work, int *lwork, int *info);
void dgeqrf_(int *M, int *N, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
void sormqr_(const char *side, const char *trans, int *m, int *n, int *k, float *A, int *lda, float *tau, float *C, int *ldc, float *work, int *lwork, int *info);
void sgesvd_(const char *jobu, const char *jobvt, int *m, int *n, float *A, int *lda, float *S, float *U, int *ldu, float *VT, int *ldvt, float *work, int *lwork, int *info);
void ssyevd_(const char *ta, const char *uplo, int *N, float *A, int *lda, float *W, float *work, int *lwork, float *iwork, int *liwork, int *info);

}


namespace DArray {
    void strrk(const char* transa, const char* transb, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float* C, int ldc);

    template<typename T>
    void multiply_matrix(const char *ta, const char *tb, T alpha, T beta, LMatrix<T> &A, LMatrix<T> &B, LMatrix<T> &C) {
        ASSERT_NOT_IMPLEMENTED;

    }
    template<>
    void multiply_matrix<float>(const char *ta, const char *tb, float alpha, float beta, LMatrix<float> &A, LMatrix<float> &B, LMatrix<float> &C) {
        if (A.dims()[1] != B.dims()[0] || A.dims()[0] != C.dims()[0] || B.dims()[1] != C.dims()[1]) {
            assert(0);
        }
        sgemm_(ta, tb, &C.dims()[0], &C.dims()[1], &A.dims()[1], &alpha, A.data(), &A.ld(),
               B.data(), &B.ld(), &beta, C.data(), &C.ld());
    }

    template<>
    void multiply_matrix<double>(const char *ta, const char *tb, double alpha, double beta, LMatrix<double> &A, LMatrix<double> &B, LMatrix<double> &C) {
        if (A.dims()[1] != B.dims()[0] || A.dims()[0] != C.dims()[0] || B.dims()[1] != C.dims()[1]) {
            assert(0);
        }
        dgemm_(ta, tb, &C.dims()[0], &C.dims()[1], &A.dims()[1], &alpha, A.data(), &A.ld(),
               B.data(), &B.ld(), &beta, C.data(), &C.ld());
    }
    template<typename T>
    void symmetric_update(const char* uplo, const char* trans, float alpha, LMatrix<T> &A12, float beta, LMatrix<T> &A22) {
        ASSERT_NOT_IMPLEMENTED;
    }
    template<>
    void symmetric_update<float>(const char* uplo, const char* trans, float alpha, LMatrix<float> &A12, float beta, LMatrix<float> &A22) {
        float none = -1, one = 1;
        ssyrk_(uplo, trans, &A22.dims()[0], &A12.dims()[0], &none, A12.data(), &A12.ld(), &one, A22.data(), &A22.ld() );
    }
    template<typename T>
    void triangular_update(const char* transa, const char* transb,  float alpha, LMatrix<T>  &A ,LMatrix<T>  &B ,float beta, LMatrix<T> &C) {
        ASSERT_NOT_IMPLEMENTED;
    }
    template<>
    void triangular_update<float>(const char* transa, const char* transb,  float alpha, LMatrix<float>  &A,LMatrix<float>  &B,  float beta, LMatrix<float> &C) {
        if(*transa != 'T' || *transb != 'N') assert(0);
        int m = C.dims()[0], n = C.dims()[1], k = B.dims()[0];
        if(abs(m-n)>1) assert(0);
        if (m==0 || n==0) return;

        if(m>n) {
            strrk(transa, transb, n,  k, alpha, A.data(), A.ld(), B.data(), B.ld(), beta, C.data(), C.ld() );
        } else if (m<n) {
            strrk(transa, transb, m,  k, alpha, A.data(), A.ld(), &B.data()[B.ld()*1], B.ld(), beta, &C.data()[C.ld()*1], C.ld() );
        } else {
            strrk(transa, transb, n,  k, alpha, A.data(), A.ld(), B.data(), B.ld(), beta, C.data(), C.ld() );
        }

    }

    void strrk(const char* transa, const char* transb, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float* C, int ldc) {
        if(n<=32) {
            sgemm_(transa, transb, &n, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
        } else {
            int m = n/2, m2 = n-m;
            strrk(transa, transb, m, k, alpha, A, lda, B, ldb, beta, C, ldc);
            sgemm_(transa, transb, &m, &m2, &k, &alpha, A, &lda, &B[m*ldb], &ldb, &beta, &C[m*ldc], &ldc);
            strrk(transa, transb, m2, k, alpha, &A[lda*m], lda, &B[m*ldb], ldb, beta, &C[m+m*ldc], ldc);
        }
    }
}