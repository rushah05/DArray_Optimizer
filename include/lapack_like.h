#pragma once
#include "darray.h"
#include "cblas_like.h"
#include "blas_like.h"
#include "utils.h"

namespace DArray {

    template<typename T>
    void cholesky_factorize(DMatrix<T> &A, char uplo, int n, int gi, int bs = 256) {
        assert(A.grid().dims()[0] == A.grid().dims()[1]);
        T one = 1, none = -1;
        int e = n+gi;
        assert(e <= A.dims()[0] && e <= A.dims()[1]);
        if (uplo == 'U' || uplo == 'u') { // only the upper triangular is referenced

            for(int i=gi; i<e; i+=bs) {
                fmt::print("at begining of iter {}, memory consumption: {:.1f}MBytes\n", i, getPeakRSS()/1.0e6);
                int b = std::min(bs, e-i);
                auto A11 = A.replicate_in_all(i, i, i+b, i+b);
                {
                    int info;
                    Tracer tracer("spotrf");
                    spotrf_("U", &b, A11.data(), &A11.ld(), &info); // fix hard coded cblas function
                    if(info != 0) {
                        fmt::print("spotrf error code {} at i= {}. Stopping\n", info, i);
                        return;
                    }
                }
                A.dereplicate_in_all(A11, i, i, i + b, i + b);
                if( i+b < e) {
                    auto A12 = A.replicate_in_columns(i, i+b, i+b, e);
                    {
                        Tracer tracer("strsm");
                        // FIXME: pick correct BLAS precision based on typename T
                        strsm_("L", "U", "T", "N", &b, &A12.dims()[1], &one, A11.data(), &A11.ld(), A12.data(),
                               &A12.ld());
                    }
                    A.dereplicate_in_columns(A12, i, i+b, i+b, e);
                    auto A21 = A.copy_upper_to_lower(A12,  i+b, e);
                    auto A22 = A.local_view(i+b, i+b, e, e);
                    if(A.grid().ranks()[0] == A.grid().ranks()[1]) {
                        DArray::Tracer t("syrk");
                        symmetric_update("U", "T", -1.0, A12, 1.0, A22);
                    } else {
                        DArray::Tracer t("trrk");
                        triangular_update("T", "N", -1, A21, A12, 1, A22);
                    }
//
                }

            }
        } else {
            ASSERT_NOT_IMPLEMENTED;
        }
    }

    template<typename T>
    void cholesky_factorize_recursive(DMatrix<T> &A, char uplo, int n, int i, int bs = 32) {
        T one = 1, none = -1;
        int e = n+i;
        assert(e <= A.dims()[0] && e <= A.dims()[1]);
        int info;
        if (uplo == 'U' || uplo == 'u') { // only the upper triangular is referenced
            if (n<=bs) { // base case
                Tracer tracer("chol_recursive_base");
                auto A11 = A.replicate_in_all(i, i, i+n, i+n);
                spotrf_("U", &n, A11.data(), &A11.ld(), &info); // fix hard coded cblas function
                A.dereplicate_in_all(A11, i, i, i + n, i + n);
            } else { // recurse
                int n1 = n / 2, n2 = n - n1;
                cholesky_factorize_recursive(A, uplo, n1, i, bs);
                DMatrixDescriptor<T> G12 {A, i, i+n1, i+n1, i+n};
                DMatrixDescriptor<T> G11 {A, i, i, i+n1, i+n1};
                DMatrixDescriptor<T> A22 {A, i+n1, i+n1, i+n, i+n};
                {
                    Tracer tracer("tri solve in recursive cholesky");
                    triangular_solve("L", "U", "T", "N", n1, n2, T(1), G11, G12, bs);
                }
                {
                    Tracer tracer("mat mul in recusive cholesky");
                    matrix_multiply("T", "N", G12, G12, A22, T(-1), T(1));
                }
                cholesky_factorize_recursive(A, uplo, n2, i+n1, bs);
            }

        } else {
            ASSERT_NOT_IMPLEMENTED;
        }
    }

    template<typename T>
    void lu_factorize_no_pivot(DMatrix<T> &A, int i, int j, int m, int n, int bs = 256) {
        assert(i+m<=A.dims()[0] && j+n<=A.dims()[1]);
        int em = i + m, en = i + n;
        std::vector<int> ipiv(m);
        float one = 1.0;
        for(; j<en && i<em; j+=bs, i+=bs) {
            int ib = std::min(bs, em-i);
            int jb = std::min(bs, en-j);
            auto A11 = A.replicate_in_all(i, j, i+ib, j+jb);
            int info;
            if(A.grid().rank()==0) A11.print("A11 before sgetrf");
            sgetrf_(&ib, &jb, A11.data(), &A11.ld(), ipiv.data(), &info);
            if(A.grid().rank()==0) {
                A11.print("A11 after sgetrf");
                fmt::print("ipiv={}\n", std::vector( ipiv.begin(), ipiv.begin()+bs));
            }
            A.dereplicate_in_all(A11, i, j, i+ib, j+jb);

            auto A21 = A.replicate_in_rows(i+ib, j, em, j+jb);

            if(A21.dims()[0] > 0 ) {
                // FIXME: pick correct BLAS precision based on typename T
                strsm_("R", "U", "N", "N",&A21.dims()[0], &jb, &one, A11.data(), &A11.ld(), A21.data(), &A21.ld());
                A.dereplicate_in_rows(A21, i+ib, j, em, j+jb);
            }
            auto A12 = A.replicate_in_columns(i, j+jb, i+ib, en);
            if(A12.dims()[1] > 0 ) {
                // FIXME: pick correct BLAS precision based on typename T
                strsm_("L","L","N","U", &ib, &A12.dims()[1], &one, A11.data(), &A11.ld(), A12.data(), &A12.ld());
                A.dereplicate_in_columns(A12, i, j+jb, i+ib, en);
            }
            if(A21.dims()[0] > 0 && A12.dims()[1] > 0) {
                auto A22 = A.local_view(i+ib, j+jb, em, en);
                multiply_matrix("N", "N", -1.0f, 1.0f, A21, A12, A22);

            }
        }
    }

    template<typename T>
    void lu_factorize_partial_pivot_replicate_panel(DMatrix<T> &A, int i, int j, int m, int n, int bs = 256) {
        assert(i+m<=A.dims()[0] && j+n<=A.dims()[1]);
        int em = i + m, en = j + n;
        std::vector<int> ipiv(bs);
        T one = 1.0;
        for(; j<en && i<em; j+=bs, i+=bs) {
            int ib = std::min(bs, em-i);
            int jb = std::min(bs, en-j);
            auto A1 = A.replicate_in_all(i, j, em, j+jb); // replicate the whole panel.
            {
                Tracer tracer("sgetrf_");
                int info;
                sgetrf_(&A1.dims()[0], &A1.dims()[1], A1.data(), &A1.ld(), ipiv.data(), &info);
                assert(info == 0);
            }
            int n_ipiv = std::min(A1.dims()[0], A1.dims()[1]);
            for(int ii=0; ii<n_ipiv; ii++) A.ipiv()[i+ii] = ipiv[ii]+i;

            A.dereplicate_in_all(A1, i, j, em, j+jb);
            A.permute_rows_ipiv(i, j+jb, em, en, ipiv, n_ipiv);

            auto A21 = A.replicate_in_rows(i+ib, j, em, j+jb);

            auto A12 = A.replicate_in_columns(i, j+jb, i+ib, en);
            if(A12.dims()[1] > 0 ) {
                Tracer trace("strsm_ & dereplicate in columns");
                // FIXME: pick correct BLAS precision based on typename T
                strsm_("L","L","N","U", &ib, &A12.dims()[1], &one, A1.data(), &A1.ld(), A12.data(), &A12.ld());
                A.dereplicate_in_columns(A12, i, j+jb, i+ib, en);
            }
            if(A21.dims()[0] > 0 && A12.dims()[1] > 0) {
                Tracer trace("sgemm");
                auto A22 = A.local_view(i+ib, j+jb, em, en);
                multiply_matrix("N", "N", -1.0f, 1.0f, A21, A12, A22);
            }
        }
    }

    template<typename T>
    void lu_factorize_partial_pivot(DMatrix<T> &A, int i, int j, int m, int n, int bs = 256) {
        Tracer tracer(__FUNCTION__ );
        assert(i+m<=A.dims()[0] && j+n<=A.dims()[1]);
        int em = i + m, en = j + n;
        T one = 1.0;
        for(; j<en && i<em; j+=bs, i+=bs) {
//            fmt::print("P[{}{}]: i{}, j{}\n", A.grid().ranks()[0], A.grid().ranks()[1], i, j);
            int ib = std::min(bs, em-i);
            int jb = std::min(bs, en-j);

            auto ipiv = lu_factorize_unblocked(A, i, j, em, j+jb);
//            return;

            int n_ipiv = ipiv.size();
//            fmt::print("ipiv={}\n", ipiv);
//            return;
            for(int ii=0; ii<n_ipiv; ii++) A.ipiv()[i+ii] = ipiv[ii]+i;
//
            A.permute_rows_ipiv(i, j+jb, em, en, ipiv.data(), n_ipiv);
            A.permute_rows_ipiv(i, 0, em, j, ipiv.data(), n_ipiv);

            auto A21 = A.replicate_in_rows(i+ib, j, em, j+jb);

            auto A12 = A.replicate_in_columns(i, j+jb, i+ib, en);
            auto A11 = A.replicate_in_all(i, j, i+ib, j+jb);
            if(A12.dims()[1] > 0 ) {
                Tracer trace("strsm_ & dereplicate in columns");
                // FIXME: pick correct BLAS precision based on typename T
                strsm_("L","L","N","U", &ib, &A12.dims()[1], &one, A11.data(), &A11.ld(), A12.data(), &A12.ld());
                A.dereplicate_in_columns(A12, i, j+jb, i+ib, en);
            }
            if(A21.dims()[0] > 0 && A12.dims()[1] > 0) {
                Tracer trace("sgemm");
                auto A22 = A.local_view(i+ib, j+jb, em, en);
                multiply_matrix("N", "N", -1.0f, 1.0f, A21, A12, A22);
            }
        }
    }
    template<typename T>
    std::vector<int> lu_factorize_unblocked(DMatrix<T> A, int i1, int j1, int i2, int j2) {
        Tracer tracer(__FUNCTION__ );
        int mn = std::min(i2-i1, j2-j1);
        std::vector<int> ipiv(mn,-1);

        int p = A.grid().ranks()[0], np = A.grid().dims()[0];
        int q = A.grid().ranks()[1], nq = A.grid().dims()[1];
        auto i12 = A.global_to_local_index({i1, i2}, p, np, 0);
        auto j12 = A.global_to_local_index({j1, j2}, q, nq, 0);

        A.grid().barrier();
        for(int j=j1, i=i1; j<j2 && i<i2; i++, j++) {
//            if(q==0) fmt::print("===== iter i={},j={} =====\n", i, j);
            int g_max_i;
            auto c_i12 = A.global_to_local_index({i, i2}, p, np, 0);
            auto c_j12 = A.global_to_local_index({j, j2}, q, nq, 0);
            // find the pivot row
            if(q == j%nq) {
                int jj = (j-q)/nq;
                int offset = jj*A.lld();
                T max_val = -1.0;
                int max_ii = -1;
                for (int ii = c_i12[0]; ii < c_i12[1]; ii++) {
                    if( fabs(A.data()[ii+offset]) > max_val ) {
                        max_val = fabs(A.data()[ii+offset]);
                        max_ii = ii;
                    }
                }
                struct {T val; int rank;} in {max_val, p}, out;
                MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MAXLOC, A.grid().comms()[0]);// FIXME: general float to T.
                g_max_i = (max_ii)*np + p;
                MPI_Bcast(&g_max_i, 1, MPI_INT, out.rank, A.grid().comms()[0]);
            }
            MPI_Bcast(&g_max_i, 1, MPI_INT, j%nq, A.grid().comms()[1]);
            ipiv[i-i1] = g_max_i - i1 + 1;
            if(i != g_max_i) { // needs to pivot
//                fmt::print("swapping {} with {}; stop swapping!\n", i, g_max_i);
                A.swap_rows(i, g_max_i, j1, j2);
            }

            if(q == j%nq) {
                T pivot = 0;
                if (p == i%np) {
                    pivot = A.data()[(i - p) / np + (j - q) / nq * A.lld()];
                }
                MPI_Bcast(&pivot, 1, convert_to_mpi_datatype<T>(), i%np, A.grid().comms()[0]);
                assert(pivot != 0);
                T inv_pivot = 1.0/pivot;
                auto i12 = A.global_to_local_index({i+1, i2}, p, np, 0);
                int jj = (j-q)/nq;
                for(int ii = i12[0]; ii < i12[1]; ii++) {
                    A.data()[ii+jj*A.lld()] *= inv_pivot;
                }
            }
            auto column = A.replicate_in_rows(i+1, j, i2, j+1); // replicate single row;
            auto row = A.replicate_in_columns(i, j+1, i+1, j2);
            auto j12 = A.global_to_local_index({j+1,j2}, q, nq, 0);
            auto i12 = A.global_to_local_index({i+1,i2}, p, np, 0);
            for(int jj = j12[0]; jj < j12[1]; jj++) {
                for(int ii = i12[0]; ii < i12[1]; ii++) {
                    A.data()[ii + jj*A.lld()] -= column[ii-i12[0]] * row[jj-j12[0]];
                }
            }

//            A.grid().barrier();
        }

        return std::move(ipiv);
    }

    // qr factorize A[i1:i2, j1:j2]; Q factor overwrites A, R factor is returned on each process
    template<typename T>
    LMatrix<T> tall_skinny_qr_factorize(DMatrix<T> A, int i1, int j1, int i2, int j2) {
        // 1. replicate in rows: rows distributed, col replicated
        auto LA = A.replicate_in_rows(i1,j1,i2,j2);
        int m = LA.dims()[0], n = LA.dims()[1];
        assert( n == j2 - j1);
        assert( m > n);

        // 2. local QR (BLAS)
        std::vector<T> tau(std::min(m,n));
        int lwork = -1, info ;
        T tmp;
        sgeqrf_( &m, &n, LA.data(), &LA.ld(), tau.data(), &tmp, &lwork, &info );
        lwork = tmp;
        std::vector<T> work(lwork);
        sgeqrf_( &m, &n, LA.data(), &LA.ld(), tau.data(), work.data(), &lwork, &info );
        assert(info == 0);
//        A.grid().print_row_replicated(LA, "R");

        // 3. stack Rs into a big tall R, size (n*np) * n, e.g. 10,000 * 100
        // FIXME: chnage datatype according to typename T.
        int np = A.grid().dims()[0];
        std::vector<T> recv(np*n*n);
        {
            LMatrix<T> R(n,n);
            copy_block(n, n, LA.data(), LA.ld(), R.data(), R.ld());
            auto err = MPI_Allgather(R.data(), n * n, MPI_FLOAT, recv.data(), n * n, MPI_FLOAT, A.grid().comms()[0]);
            assert(err == MPI_SUCCESS);
        }

        LMatrix<T> Rs(np*n, n);
        int ldRs = Rs.ld();
        for(int b=0; b<np; b++) {
            T* recvblock = &recv.data()[n*n*b];
            for(int j=0; j<n; j++) {
                for (int i=0; i<n; i++)
                    Rs.data()[b*n+i+j*ldRs] = (j<i)? 0 : recvblock[i + j*n];
            }
        }

        // 4. qr the stacked R
        LMatrix<T> Q(np*n, n);
        int ldq = Q.ld();
        for (int i=0; i<ldq; i++) {
            for (int j = 0; j < n; j++)
                Q.data()[i + j * ldq] = (i == j) ? 1.0 : 0;
        }
        {
            int m = np*n;
            std::vector<T> tau(n);
            int nwork, lwork = -1, info;
            T tmp;
            sgeqrf_(&m, &n, Rs.data(), &Rs.ld(), tau.data(), &tmp, &lwork, &info);
            lwork = tmp;
            std::vector<T> work(lwork);
            sgeqrf_(&m, &n, Rs.data(), &Rs.ld(), tau.data(), work.data(), &lwork, &info);
            assert(info == 0);

            // Form the explict Q
            // FIXME: use sorgqr instead of sormqr ?
            lwork = -1;
            sormqr_("L", "N", &m, &n, &n, Rs.data(), &Rs.ld(), tau.data(), Q.data(), &Q.ld(), &tmp, &lwork, &info);
            lwork = tmp;
            work.resize(lwork);
            sormqr_("L", "N", &m, &n, &n, Rs.data(), &Rs.ld(), tau.data(), Q.data(), &Q.ld(), work.data(), &lwork, &info);
            assert(info == 0);
        }

        // 5. post-multiply the Q of stacked R
        // LA Q has size m*n; R Q has size n*n
        // LA Q * R Q -> m*n
        // (I - 2*tau*H[1]*H[1]') m*m apply to n*n -> m*n
        LMatrix<T> RQ(m, n);
        auto pi = A.grid().ranks()[0];
        int ldRQ = RQ.ld(), ldQ = Q.ld();
        for(int j=0; j<n; j++) {
            for(int i=0; i<n; i++) {
                RQ.data()[i+j*ldRQ] = Q.data()[n*pi + i + j*ldQ];
            }
            for(int i=n; i<m; i++)
                RQ.data()[i+j*ldRQ] = 0;
        }
        lwork = -1;
        sormqr_("L", "N", &m, &n, &n, LA.data(), &LA.ld(), tau.data(), RQ.data(), &RQ.ld(), &tmp, &lwork, &info);
        lwork = tmp;
        work.resize(lwork);
        sormqr_("L", "N", &m, &n, &n, LA.data(), &LA.ld(), tau.data(), RQ.data(), &RQ.ld(), work.data(), &lwork, &info);
        assert(info == 0);

        // 6. write Q back to distributed A
        auto grid = A.grid();
//        grid.print(RQ, "RQ");
        A.dereplicate_in_rows(RQ, i1, j1, i2, j2);

        LMatrix<T> R(n,n);
        copy_block(n, n, Rs.data(), Rs.ld(), R.data(), R.ld());
        return R;
    }
    
    template<typename T>
    void qr_factorize_unblocked(DMatrix<T> A, int i1, int j1, int i2, int j2) {

    }



    template<typename T>
    LMatrix<T> svd(int rank, LMatrix<T> LA, int i1, int j1, int i2, int j2) {
        int m = LA.dims()[0], n = LA.dims()[1];
        assert (m == n);
        
        DArray::LMatrix<T> S(m,1);
        DArray::LMatrix<T> U(m, n);
        DArray::LMatrix<T> VT(n, n);

        int info=-1, lwork = -1;
        T tmp;
        //perform a svd on Q^T*A
        sgesvd_("A", "A", &m, &n, LA.data(), &LA.ld(), S.data(), U.data(), &U.ld(), VT.data(), &VT.ld(), &tmp, &lwork, &info);
        assert(info == 0);
        lwork = tmp;
        DArray::LMatrix<T> work(lwork, 1);

        info=-1;
        sgesvd_("A", "A", &m, &n, LA.data(), &LA.ld(), S.data(), U.data(), &U.ld(), VT.data(), &VT.ld(), work.data(), &lwork, &info);
        assert(info == 0);

        // DArray::LMatrix<T> diagS(m, n);
        // diagS.diag(S);
        // S.save_to_file("S.csv");
        return S;
    }
}
