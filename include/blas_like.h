#pragma once
#include "darray.h"
#include "cblas_like.h"

namespace DArray {



    template<typename T>
    void triangular_triangular_multiply(DMatrix<T>& A,  DMatrix<T>& B,  DMatrix<T>& C,  T alpha, T beta) {
        Tracer tracer(__FUNCTION__ );
        assert(A.dims()[1] == B.dims()[0] && C.dims()[0] == A.dims()[0] && C.dims()[1] == B.dims()[1]);
        int bs = 128;
        int m = C.dims()[0], n = C.dims()[1];
        assert(m==n);
        int np = C.grid().dims()[0], nq = C.grid().dims()[1];
        int p = C.grid().ranks()[0], q = C.grid().ranks()[1];
        for (int j=0; j<n; j+=bs) {
//            fmt::print("tri_tri_mul: iter {}\n", j);
            int i = j;
            int jb = std::min(bs, n-j);
            int ib = jb;
            auto A1 = A.replicate_in_rows(i, j, m, j+jb);
            auto i12 = C.global_to_local_index({i, i+ib}, p, np, 0);
            auto j12 = C.global_to_local_index({j, j+jb}, q, nq, 0);
            for(int ii=i12[0]; ii<i12[1]; ii++) {
                int gii = ii*np + p;
                for(int gjj=j; gjj<j+jb ; gjj++) {
                    if(gjj>gii) A1(ii-i12[0], gjj-j) = 0;
                    else if(gjj==gii) A1(ii-i12[0], gjj-j) = 1;
                }
            }

            A.grid().for_each_process_in_column([&]() {
//                if(q==0) A1.print(fmt::format("P[{},{}]: A1", p, q));
            });
            auto B1 = B.replicate_in_columns(j, i, j+jb, n);
            for(int gii=i; gii<i+ib; gii++) {
                for(int jj=j12[0]; jj<j12[1] ; jj++) {
                    int gjj = jj*nq + q;
                    if(gjj<gii) {
                        B1(gii-i, jj-j12[0]) = 0;
                    }

                }
            }
            A.grid().for_each_process_in_row([&]() {
//                if(p==0) B1.print(fmt::format("P[{},{}]: B1", p, q));
            });
//            B1.print("B1");
            auto C1 = C.local_view(i,j,m, n);
//            C1.print("C1");
            if(j==0)
                multiply_matrix("N", "N", alpha, beta, A1, B1, C1);
            else
                multiply_matrix("N", "N", alpha, 1.0f, A1, B1, C1);
//            return;
        }

    }

    template<typename T>
    void matrix_multiply(const char* transa, const char* transb, DMatrixDescriptor<T> Adesc, DMatrixDescriptor<T> Bdesc,
                         DMatrixDescriptor<T> Cdesc, T alpha, T beta, int bs = 128) {
        Tracer tracer(__FUNCTION__ );
        if ((*transa == 'T' || *transa == 't') && (*transb == 'N' || *transa == 'n')){
            assert(Adesc.n() == Cdesc.m() && Bdesc.n() == Cdesc.n());
//            fmt::print("matmul: m={}, n={}, k={}\n", Cdesc.m(), Cdesc.n(), Adesc.m());
            // FIXME: add alignment assert:
            int k = Adesc.m();
            // algorithm 1: suitable for outer product:
            auto& grid = Adesc.matrix.grid();
            for (int kk = 0; kk < k; kk += bs) {
                int ke = std::min(k, kk + bs);

                auto A = Adesc.matrix.transpose_and_replicate_in_rows(Adesc.i1 + kk, Adesc.j1, Adesc.i1 + ke, Adesc.j2);
                auto B = Bdesc.matrix.replicate_in_columns(Bdesc.i1 + kk, Bdesc.j1, Bdesc.i1 + ke, Bdesc.j2);
                auto C = Cdesc.matrix.local_view(Cdesc.i1, Cdesc.j1, Cdesc.i2, Cdesc.j2);
                assert(C.dims()[0] == A.dims()[0] && C.dims()[1] == B.dims()[1] && A.dims()[1] == B.dims()[0]);
                if(kk > 0) {
                    beta = 1;
                }
                // FIXME: don't hardcode type
                {
                    Tracer tracer("sgemm in matrix_multiply");
                    sgemm_("N", "N", &C.dims()[0], &C.dims()[1], &A.dims()[1], &alpha, A.data(), &A.ld(), B.data(),
                           &B.ld(), &beta, C.data(), &C.ld());
                }

            }
        }

    }

    template<typename T>
    void triangular_solve(const char* side, const char* uplo, const char* transa, const char* diag,
                          int m, int n, T alpha, DMatrixDescriptor<T> Adesc, DMatrixDescriptor<T> Bdesc, int bs = 64) {
        T one = 1;
        auto grid = Adesc.matrix.grid();
        if ((*uplo == 'U' || *uplo == 'u') && (*transa == 'T' || *transa == 't') && (*side == 'L' || *side == 'l')){
            assert(Adesc.n() == Bdesc.m());
            assert(Adesc.j1 % grid.dims()[1] == Bdesc.i1 % grid.dims()[0] );
            if (m <= bs) {
                Tracer tracer("tri solve base");
                auto A = Adesc.matrix.replicate_in_all(Adesc.i1, Adesc.j1, Adesc.i2, Adesc.j2);
                A.clear_lower();
                auto B = Bdesc.matrix.replicate_in_columns(Bdesc.i1, Bdesc.j1, Bdesc.i2, Bdesc.j2);
                assert(A.dims()[1] == B.dims()[0]);
                // FIXME: don't hardcode strsm_
                strsm_("L", "U", "T", diag, &B.dims()[0], &B.dims()[1], &one, A.data(), &A.ld(), B.data(), &B.ld());
                Bdesc.matrix.dereplicate_in_columns(B, Bdesc.i1, Bdesc.j1, Bdesc.i2, Bdesc.j2);
            } else {
                int m1 = m / 2, m2 = m - m1;
                DMatrixDescriptor<T> A22 {Adesc.matrix, Adesc.i1 + m1, Adesc.j1 + m1, Adesc.i2, Adesc.j2};
                DMatrixDescriptor<T> B2 {Bdesc.matrix, Bdesc.i1 + m1, Bdesc.j1, Bdesc.i2, Bdesc.j2};
                DMatrixDescriptor<T> B1 {Bdesc.matrix, Bdesc.i1, Bdesc.j1, Bdesc.i1 + m1, Bdesc.j2};
                DMatrixDescriptor<T> A12 {Adesc.matrix, Adesc.i1, Adesc.j1+m1, Adesc.i1+m1, Adesc.j2};
                DMatrixDescriptor<T> A11 {Adesc.matrix, Adesc.i1 , Adesc.j1, Adesc.i1 + m1, Adesc.j1+m1};

                triangular_solve(side, uplo, transa, diag, m1, n, alpha, A11, B1);
                matrix_multiply("T", "N", A12, B1, B2, T(-1), T(1) );
                triangular_solve(side, uplo, transa, diag, m2, n, alpha, A22, B2);
            }
        } else {
            ASSERT_NOT_IMPLEMENTED;
        }

    }
}