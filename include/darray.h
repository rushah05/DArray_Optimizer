#pragma once

#include <mpi.h>
#include <cassert>
#include <random>
#include <cstdio>
#include <chrono>
#include <functional>
#include <memory>
#include <iostream>
#include <chrono>
#define FMT_HEADER_ONLY 1
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "utils.h"
#include <cmath>

#define ASSERT_NOT_IMPLEMENTED assert(0)

namespace DArray {
    template<typename T>
    void copy_block(int m, int n, T* src, int ldsrc, T* dest, int lddest) {
//            Tracer tracer("copy_block");

        for(int j=0; j<n; j++) {
            int lj1 = lddest*j, lj2 = ldsrc*j;
            for(int i=0; i<m; i++) {
                dest[i+lj1] = src[i+lj2];
            }
        }
    }

    // dest = src^T
    // src: m*n.
    // dest: n*m
    template<typename T>
    void copy_transposed_block(int m, int n, T* src, int ldsrc, T* dest, int lddest) {
        for(int j=0; j<n; j++) {
            int lj1 = lddest*j, lj2 = ldsrc*j;
            for(int i=0; i<m; i++) {
                dest[j+i*lddest] = src[i+lj2];
            }
        }
    }


    template<typename T>
    class LMatrix;
    template<typename T>
    class DMatrix;
    class Grid;

    enum Layout { RowMajor, ColMajor };

    template<typename T>
    MPI_Datatype convert_to_mpi_datatype() {
        return MPI_BYTE;
    }

    template<>
    MPI_Datatype convert_to_mpi_datatype<float>() {
        return MPI_FLOAT;
    }

    inline int size_of_local_part(int globalsize, int p, int np) {
        return (globalsize - p + np -1 ) / np;
    }

    //
    inline std::vector<int> list_local_part_sizes(int n, int np) {
        std::vector<int> l(np, 0);
        for(int i=0; i<np; i++) {
            l[i] = size_of_local_part(n, i, np);
        }
        return l;
    }

    inline std::vector<int> scan_local_part(int n, int np) {
        std::vector<int> l(np, 0);
        for(int i=1; i<np; i++) {
            l[i] = size_of_local_part(n, i - 1, np) + l[i - 1];
        }
        return l;
    }




    class Grid {
        std::array<int,2> m_dims;
        std::array<int,2> m_ranks;
        std::array<MPI_Comm,2> m_comms; // row comm, col comm
        MPI_Comm m_comm;
        int m_np;
        int m_rank;
        Layout m_layout;
    public:
        // given global comm, create p*q grid.
        Grid(const MPI_Comm &comm, int p, int q) : m_comm{comm}, m_dims{p, q}, m_layout{ColMajor} {
            MPI_Comm_size(comm, &m_np);
            assert(p != 0 && q != 0 && p * q <= m_np);

            MPI_Comm_rank(comm, &m_rank);
            if (m_layout == ColMajor) {
                m_ranks[0] = m_rank % p;
                m_ranks[1] = m_rank / p;
                int myrank;
                auto err1 = MPI_Comm_split(comm, m_rank / p, m_rank % p, &m_comms[0]); // col comm
                assert(err1 == MPI_SUCCESS);
                MPI_Comm_rank(m_comms[0], &myrank);
                assert(myrank == m_ranks[0]);
                auto err2 = MPI_Comm_split(comm, m_rank % p, m_rank / p, &m_comms[1]); // row comm
                assert(err2 == MPI_SUCCESS);
                MPI_Comm_rank(m_comms[1], &myrank);
                assert(myrank == m_ranks[1]);
            }
        }

        int rank() { return m_rank; }

        std::array<int,2> ranks() { return m_ranks; }

        int np() { return m_np; }

        std::array<int,2>  dims() { return m_dims; }

        MPI_Comm comm() { return m_comm; }

        std::array<MPI_Comm,2> comms() { return m_comms; }

        Layout layout() { return m_layout; }

        int convert_to_linear(int i, int j) {
            if( m_layout == ColMajor) {
                return i+j*m_dims[0];
            } else {
                ASSERT_NOT_IMPLEMENTED;
		        return 0;
            }
        }
        std::array<int,2> convert_to_pq(int p) {
            if(layout() == ColMajor) {
                int np = dims()[0];
                return {p%np, p/np};
            } else {
                ASSERT_NOT_IMPLEMENTED;
                return {};
            }
        }

        void barrier() {
            MPI_Barrier(comm());
        }

        void barrier_col() {
            MPI_Barrier(comms()[0]);
        }

        void barrier_row() {
            MPI_Barrier(comms()[1]);
        }

        static void np_to_pq(int np, int &p, int &q) {
            int k = round(sqrt(np));
            for(; k>0; k--) {
                if (np%k==0) {
                    p = k;
                    q = np/k;
                    break;
                }
            }
        }

        void for_each_process_in_column(std::function<void(void)> func) {
            for(int pi = 0; pi<dims()[0]; pi++) {
                if(ranks()[0] == pi) {
                    func();
                }
                barrier_col();
            }
        }

        void for_each_process_in_row(std::function<void(void)> func) {
            for(int qi = 0; qi<dims()[1]; qi++) {
                if(ranks()[1] == qi) {
                    func();
                }
                barrier_row();
            }
        }

        void for_each_process_in_grid(std::function<void(void)> func) {
            barrier();
            for(int pi = 0; pi<dims()[0]; pi++) {
                for(int pj = 0; pj<dims()[1]; pj++) {
                    if (ranks()[0] == pi && ranks()[1] == pj) {
                        func();
                    }
                    barrier();
                }
            }
            barrier();
        }
        template<typename T>
        void print_row_replicated(LMatrix<T> A, const std::string &title) {
            barrier();
            if(ranks()[1] == 0) {
                for (int pi = 0; pi < dims()[0]; pi++) {
                    if (ranks()[0] == pi) {
                        A.print(fmt::format("P[{},*] {}", pi, title));
                    }
                    barrier_col();
                }
            }
            barrier();
        }
        template<typename T>
        void print_column_replicated(LMatrix<T> A, const std::string &title) {
            barrier();
            if(ranks()[0] == 0) {
                for (int pi = 0; pi < dims()[1]; pi++) {
                    if (ranks()[1] == pi) {
                        A.print(fmt::format("P[*,{}] {}", pi, title));
                    }
                    barrier_row();
                }
            }
            barrier();
        }
        template<typename T>
        void print(LMatrix<T> A, const std::string &title) {
            barrier();
            for(int pi = 0; pi<dims()[0]; pi++) {
                for(int pj = 0; pj<dims()[1]; pj++) {
                    if (ranks()[0] == pi && ranks()[1] == pj) {
                        A.print(fmt::format("P[{},{}] {}", pi,pj,  title));
                    }
                    barrier();
                }
            }
            barrier();
        }
        //



    };

    inline std::ostream &operator<<(std::ostream &os, Grid &grid) {
        if (grid.rank() == 0)
            return os << "Grid: np=" << grid.np() << " layout " << grid.layout() << std::endl;
        else {
            return os << "ranks: [" << grid.ranks()[0] << ", " << grid.ranks()[1] << "]" << std::endl;
        }
    }

    template<typename T>
    class LMatrix { // local matrix
        static size_t total_allocated_bytes;
        static size_t maximum_allocated_bytes;
        std::shared_ptr<T[]> m_storage;
        T* m_data;
        std::array<int,2> m_dims;
        Layout layout;
        int m_ld; // leading dimension.
    public:
        LMatrix() : m_dims{0, 0}, layout{ColMajor}, m_ld{0}, m_data{} {}
        LMatrix(int m, int n) : m_dims{m, n}, layout{ColMajor} {
//            fmt::print("new {}\n", m*n);
            if(m*n<0) {
                fmt::print(Backtrace());
            }
            std::shared_ptr<T[]> s(new T[m*n]);
            m_storage = s;
            m_ld = m;
            m_data = m_storage.get();


        }
        LMatrix(int m, int n, int ld) : m_dims{m, n}, layout{ColMajor} {
            std::shared_ptr<T[]> s(new T[ld*n]);
            m_storage = s;
            m_data = m_storage.get();
            m_ld = ld;


        }
        LMatrix(int m, int n, int ld, T* buffer)
            : m_dims{m, n}, layout{ColMajor},  m_ld{ld}, m_data{buffer}, m_storage{nullptr} {
        }

//        ~LMatrix() {
//
//        }

        T &operator[](int i) {return m_data[i];}
        T operator[](int i) const {return m_data[i];}
        T &operator()(int i,int j) {return m_data[i+j*m_ld];}
        T operator()(int i, int j) const {return m_data[i+j*m_ld];}
        int& ld() {return m_ld; }
        void set_ld(int ld_) {m_ld = ld_; }
        T* data() {return m_data;}
        std::array<int,2> dims() {return m_dims;}

        void print(const std::string title) {
            std::cout << title << std::endl;
            for(int i=0; i < m_dims[0]; i++) {
                for(int j=0; j < m_dims[1]; j++) {
                    printf("%8.3f ", (*this)(i,j));
                }
                printf("\n");
            }
        }
        void save_to_file(const std::string filepath) {
            FILE *f = fopen(filepath.c_str(), "w");
            assert(f);
            for(int i=0; i < m_dims[0]; i++) {
                for(int j=0; j < m_dims[1]; j++) {
                    fmt::print(f, "{}, ", (*this)(i,j));
                }
                fmt::print(f, "\n");
            }
            fclose(f);
        }

        void set_constant(T a) {
            for (int i = 0; i < m_dims[0]; i++) {
                for (int j = 0; j < m_dims[1]; j++) {
                    (*this)(i,j) = a;
                }
            }
        }

        void set_Identity() {
            for (int i = 0; i < m_dims[0]; i++) {
                for (int j = 0; j < m_dims[1]; j++) {
                    if(i == j) (*this)(i,j) = 1.0;
                    else (*this)(i,j) = 0.0;
                }
            }
        }

        void diag(LMatrix<T> E) {
            for (int i = 0; i < m_dims[0]; i++) {
                for (int j = 0; j < m_dims[1]; j++) {
                    if(i == j) (*this)(i,j) = E.data()[i];
                    else (*this)(i,j) = 0.0;
                }
            }
        }

        void set_function(std::function<T (int, int)> func) {
            for (int i = 0; i < dims()[0]; i++) {
                for (int j = 0; j < dims()[1]; j++) {
                    (*this)(i,j) = func(i, j);
                }
            }
        }

        LMatrix<T> transpose() {
            int m = m_dims[0], n = m_dims[1];
            LMatrix<T> At(n, m);
            copy_transposed_block(m, n, m_data, m_ld, At.data(), At.ld());
            return At;
        }

        void clear_upper() {
            int m = m_dims[0], n = m_dims[1];
            for(int j=0; j<n; j++) {
                for(int i=0; i<j; i++)
                    m_data[i+j*m_ld] = 0;
            }
        }

        void clear_lower() {
            int m = m_dims[0], n = m_dims[1];
            for(int j=0; j<n; j++) {
                for(int i=j+1; i<m; i++)
                    m_data[i+j*m_ld] = 0;
            }
        }



    };


    template<typename T>
    class DMatrix {
        Grid m_grid;
        std::array<int,2> m_dims;
        std::array<int,2> m_ldims;

        std::shared_ptr<T[]> m_storage;
//        std::vector<T> m_storage;
        T *m_data;
        int m_lld; // local leading dimension
//        Layout m_layout;
        T unused;
        std::vector<int> m_permutation;
        std::vector<int> m_ipiv;
    public:
        // allocating and owns memory
        DMatrix(Grid &grid, int m, int n)
                : m_grid{grid}, m_dims{m, n}
                , m_permutation{std::move(std::vector(m, -1))}
                , m_ipiv{std::move(std::vector<int>(m, -1))} {
            auto r = grid.ranks();
            auto s = grid.dims();
            m_ldims[0] = size_of_local_part(m, r[0], s[0]);
            m_ldims[1] = size_of_local_part(n, r[1], s[1]);
            m_lld = m_ldims[0];
            std::shared_ptr<T[]> ss(new T[m_ldims[0] * m_ldims[1]]);
            m_storage = ss;
            m_data = m_storage.get();
//            fmt::print("DMatrix: allocating {}MB\n", 1.0*m_ldims[0] * m_ldims[1] * sizeof(T)/1.0e6 );
        }

        // attaching to a buffer. The buffer is incomplete.
        DMatrix(Grid &grid, int m, int n, T* buf, int ld, int i1, int j1)
            : m_grid{grid}, m_dims{m,n}
            , m_permutation{}
            , m_ipiv{}
            , m_storage{}
            , m_data {nullptr}
            , m_ldims{0,0}
            , m_lld{ld}{
            auto i12 = global_to_local_index({i1, m}, grid.ranks()[0], grid.dims()[0], 0);
            auto j12 = global_to_local_index({j1, n}, grid.ranks()[1], grid.dims()[1], 0);
            m_data = buf - (i12[0]+j12[0]*ld);
        }


        DMatrix<T> clone() {
            auto A(*this);
            std::shared_ptr<T[]> ss(new T[A.m_ldims[0] * A.m_ldims[1]]);
            memcpy(ss.get(), m_storage.get(), sizeof(T)*A.m_ldims[0] * A.m_ldims[1]);
            A.m_storage = std::move(ss);
            A.m_data = A.m_storage.get();
            return A;
        }

        std::array<int, 2> dims() { return m_dims;}
        std::array<int, 2> ldims() { return m_ldims;}
        Grid& grid() {return m_grid; }
//        T &operator()(int i, int j) {return m_data[i + j * m_lld];}
//        T operator()(int i, int j) const {return m_data[i + j * m_lld];}
        T* data() {return m_data; }
        int lld() {return m_lld; }
        std::vector<int>& ipiv() {return m_ipiv; }


         //global row,col indexing into the matrix; only the possessing process operate.
        inline T& at(int i, int j) {
            int np = grid().dims()[0], nq = grid().dims()[1];
            int p = i % np, q = j % nq;
            if(p == grid().ranks()[0] && q == grid().ranks()[1]) {
                return m_data[(i-p)/np + m_lld * (j-q)/nq];
            } else {
                return unused;
            }
        }

        inline T operator()(int i, int j)  {
            int np = grid().dims()[0], nq = grid().dims()[1];
            int p = i % np, q = j % nq;
            int root = grid().convert_to_linear(p,q);
            T val;
            if(grid().rank() == root)  {
                val = m_data[(i-p)/np + m_lld * (j-q)/nq];
            }
            MPI_Bcast(&val, 1, convert_to_mpi_datatype<T>(), root, grid().comm());
            return val;
        }


        void set_uniform(T a, T b) {
            std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
            std::uniform_real_distribution<T> distribution(a, b);
            for (int i = 0; i < m_ldims[0]; i++) {
                for (int j = 0; j < m_ldims[1]; j++) {
                    m_data[i+j*m_lld] = distribution(generator);

                }
            }
        }

        void set_normal(T mean, T stddev) {
            std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
            std::normal_distribution<T> distribution(mean, stddev);
            for (int i = 0; i < m_ldims[0]; i++) {
                for (int j = 0; j < m_ldims[1]; j++) {
//                    (*this)(i, j) = distribution(generator);
                    m_data[i+j*m_lld] = distribution(generator);
                }
            }
        }

        void set_normal_seed(T mean, T stddev, int seed) {
            std::mt19937 generator(seed);
            std::normal_distribution<T> distribution(mean, stddev);
            for (int i = 0; i < m_ldims[0]; i++) {
                for (int j = 0; j < m_ldims[1]; j++) {
//                    (*this)(i, j) = distribution(generator);
                    m_data[i+j*m_lld] = distribution(generator);
                }
            }
        }

        void set_constant(T a) {
            for (int i = 0; i < m_ldims[0]; i++) {
                for (int j = 0; j < m_ldims[1]; j++) {
                    //(*this)(i, j) = a;
                    m_data[i+j*lld()] = a;
                }
            }
        }

        void set_identity() {
            auto d = m_grid.dims();
            auto r = m_grid.ranks();
            for (int i = 0; i < m_ldims[0]; i++) {
                int gi = d[0] * i + r[0];
                for (int j = 0; j < m_ldims[1]; j++) {
                    int gj = d[1] * j + r[1];
                    if (gi==gj) (*this)(i, j) = 1;
                    else (*this)(i,j) = 0;
                }
            }
        }

        void set_value_from_matrix(T* x, int ldx) {
            auto d = m_grid.dims();
            auto r = m_grid.ranks();
            for (int i = 0; i < m_ldims[0]; i++) {
                int gi = d[0] * i + r[0];
                for (int j = 0; j < m_ldims[1]; j++) {
                    int gj = d[1] * j + r[1];
                    // if(m_grid.rank() == 3) printf("gi::%d, gj::%d \n", gi, gj);
                    m_data[i+j*m_lld] = x[gi+gj*ldx];
                }
            }
        }

        void set_function(std::function<T (int, int)> func) {
            auto d = m_grid.dims();
            auto r = m_grid.ranks();
            for (int i = 0; i < m_ldims[0]; i++) {
                int gi = d[0] * i + r[0];
                for (int j = 0; j < m_ldims[1]; j++) {
                    int gj = d[1] * j + r[1];
                    m_data[i+j*m_lld] = func(gi,gj);
                }
            }
        }


        void to_double(DMatrix<double> E) {
            for (int i = 0; i < m_ldims[0]; i++) {
                for (int j = 0; j < m_ldims[1]; j++) {
                    E.data()[i+j*lld()] = (*this)(i, j);
                }
            }
        }


        // forbenius norm
        double fnorm() {
            double sqr_sum = 0;
            for(int i=0; i<ldims()[0]; i++) {
                for(int j=0; j<ldims()[1]; j++) {
                    sqr_sum += m_data[i+j*m_lld] * m_data[i+j*m_lld];
                }
            }
            double  all_sum = 0;
            MPI_Allreduce(&sqr_sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, grid().comm());
            return sqrt(all_sum);
        }

        void print_by_process(const std::string &title)  {

            auto d = m_grid.dims();
            for(int pi=0; pi<grid().np(); pi++) {
                if(grid().rank() == pi){
                    if (title != "") {
                        printf("P[%3d,%3d]: ", m_grid.ranks()[0], m_grid.ranks()[1]);
                        std::cout << title << std::endl;
                    }
                    auto r = m_grid.ranks();
                    for (int i = 0; i < m_ldims[0]; i++) {
                        for (int j = 0; j < m_ldims[1]; j++) {
                            int gi = d[0] * i + r[0];
                            int gj = d[1] * j + r[1];
                            printf("[%2d,%2d] %8.3f  ", gi, gj, (*this)(i, j));
                        }
                        printf("\n");
                    }
                }
                grid().barrier();
            }
        }

        // destructive resize.
//        void resize(int m, int n) {
//            if (m==m_dims[0] && n==m_dims[1])
//                return;
//            m_dims[0] = m; m_dims[1] = n;
//            auto r = m_grid.ranks();
//            auto s = m_grid.dims();
//            m_ldims[0] = (m - r[0] + s[0] - 1) / s[0];
//            m_ldims[1] = (n - r[1] + s[1] - 1) / s[1];
//            m_lld = m_ldims[0];
//            delete[] m_data;
//            m_data = new T[m_ldims[0] * m_ldims[1]];
//        }


        // collect DMatrix on a single process "root"; allocates memory.
        LMatrix<T>  collect(int root) {
            Tracer tracer(__FUNCTION__ );
            if(root<0 || root>m_grid.np())
                assert(0);
            int np = m_grid.dims()[0];
            int nq = m_grid.dims()[1];
            int m = m_dims[0], n = m_dims[1];

            if (m_grid.rank() == root) {
                auto buffer = DArray::LMatrix<T>(m, n);
                int maxmm = (m+np-1)/np, maxnn = (n+nq-1)/nq;
                auto recvbuf = DArray::LMatrix<T>(maxmm, maxnn);
                for(int pi = 0; pi < np; pi++) {
                    for(int qi = 0; qi < nq; qi++) {
                        int mm = size_of_local_part(m, pi, np);
                        int nn = size_of_local_part(n, qi, nq);
                        recvbuf.set_ld(mm);
                        int src = m_grid.convert_to_linear(pi, qi);
                        if (src == root) {
                            copy_block(mm, nn, data(), lld(), recvbuf.data(), recvbuf.ld());
                        } else {
                            MPI_Status status;
                            int err = MPI_Recv(recvbuf.data(), mm * nn, convert_to_mpi_datatype<T>(), src, pi + qi, m_grid.comm(), &status);
                            assert(err == MPI_SUCCESS);
                        }
                        scatter_local_to_global(mm, nn, recvbuf.data(), recvbuf.ld(), buffer.data(), buffer.ld(), pi, qi, np, nq);

                    }
                }
                return buffer;
            } else {
                if (lld() == ldims()[0]) {
                    int err = MPI_Send(m_data, m_ldims[0] * m_ldims[1], convert_to_mpi_datatype<T>(), root,
                                       m_grid.ranks()[0] + m_grid.ranks()[1], m_grid.comm());
                    assert(err == MPI_SUCCESS);
                    return LMatrix<T>();
                } else {
                    ASSERT_NOT_IMPLEMENTED;
                    return LMatrix<T>();
                }
            }
        }



        void collect_and_print(const std::string &title, int root = 0) {
            auto local_matrix = collect(root);
            if(grid().rank()==root) {
                local_matrix.print(title);
            }
        }

        void collect_and_print(const std::string &title, int i1, int j1, int i2, int j2) {
            auto A = replicate_in_all(i1, j1, i2, j2);
            if(grid().rank()==0) {
                A.print("title");
            }
        }

        // scatter a contiguous buffer to global matrix
        void scatter_local_to_global(int m, int n, T* src, int ldsrc, T* dest, int lddest, int p, int q, int np, int nq) {
            for(int i=0; i<m; i++) {
                for(int j=0; j<n; j++) {
                    dest[(i*np+p) + (j*nq+q)*lddest] = src[i+j*ldsrc];
                }
            }
        }

        // local part of global submatrix A[i1:i2, j1:j2]
        LMatrix<T> local_view(int i1, int j1, int i2, int j2) {
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//            printf("R%3d: i12={%d,%d}, j12={%d,%d}\n", grid().rank(), i12[0], i12[1], j12[0], j12[1]);
            int mm = i12[1]-i12[0], nn = j12[1]-j12[0];

            return LMatrix<T>(mm, nn, lld(), &data()[i12[0]+j12[0]*lld()]);
        }

        // replicate submatrix at A[i1:i2, j1:j2], left inclusive, right exclusive
        LMatrix<T> replicate_in_all(int i1, int j1, int i2, int j2) {
            Tracer tracer(__FUNCTION__ );
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
            int mm = i12[1] - i12[0];
            int nn = j12[1] - j12[0];

            LMatrix<T> result(i2-i1, nn);
            {             // stage 1: replicate in columns;
                LMatrix<T> sendbuf(mm, nn);
                copy_block(mm, nn, &data()[i12[0] + j12[0] * lld()], lld(), sendbuf.data(), sendbuf.ld());
                auto block_sizes = list_local_part_sizes(i1, i2, grid().dims()[0]);
                LMatrix<T> recvbuf(i2 - i1, nn);
                std::vector<int> displacements(grid().dims()[0], 0);
                std::vector<int> recv_counts(grid().dims()[0], 0);

                for (int i = 1; i < displacements.size(); i++) {
                    displacements[i] = displacements[i - 1] + block_sizes[i - 1] * nn;
                    recv_counts[i] = block_sizes[i] * nn;
                }
                recv_counts[0] = block_sizes[0] * nn;
#ifdef DEBUG_REPLICATE
                if (grid().ranks()[0] == 0) {
                    fmt::print("Stage 1: P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1], recv_counts,
                               displacements);
                } else {
//                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
                }
#endif

                {
                    Tracer tracer("Allgatherv in stage 1");
                    int err = MPI_Allgatherv(sendbuf.data(), mm * nn, convert_to_mpi_datatype<T>(),
                                             recvbuf.data(), recv_counts.data(), displacements.data(),
                                             convert_to_mpi_datatype<T>(),
                                             grid().comms()[0]);
                    assert(err == MPI_SUCCESS);
                }

                {
                    auto C = result.data(), ldc = result.ld(), A = recvbuf.data();
                    Tracer tracer("scatter in stage 1 in replicate_in_all");
                    for (int pi = 0; pi < grid().dims()[0]; pi++) {
                        auto i12 = global_to_local_index({i1, i2}, pi, grid().dims()[0], 0);
                        int mm = i12[1] - i12[0];
                        int dpi = displacements[pi];
                        int np = grid().dims()[0];
                        for (int j = 0; j < nn; j++) {
                            int jmm = j*mm;
                            for (int i = 0; i < mm; i++) {
                                int gi = (i + i12[0]) * np + pi;
//                                result(gi - i1, j) = recvbuf.data()[dpi + i + jmm];
                                C[gi-i1+j*ldc] = A[dpi+i+jmm];
                            }
                        }
                    }
                }
            }

            LMatrix<T> allresult(i2-i1, j2-j1);
            {// stage 2: replicate in rows:
                mm = i2 - i1;
                auto block_sizes = list_local_part_sizes(j1, j2, grid().dims()[1]);
                LMatrix<T> recvbuf(mm, j2-j1);
                std::vector<int> displacements( grid().dims()[1], 0 );
                std::vector<int> recv_counts( grid().dims()[1], 0);

                for(int i=1; i<displacements.size(); i++) {
                    displacements[i] = displacements[i-1] + block_sizes[i-1]*mm;
                    recv_counts[i] = block_sizes[i]*mm;
                }
                recv_counts[0] = block_sizes[0]*mm;
#ifdef DEBUG_REPLICATE
                if(grid().ranks()[1]==0) {
                    fmt::print("Stage 2: P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1],recv_counts, displacements );
                } else {
//                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
                }
#endif
                {
                    Tracer tracer("Allgatherv in stage 2");
                    int err = MPI_Allgatherv(result.data(), mm * nn, convert_to_mpi_datatype<T>(),
                                             recvbuf.data(), recv_counts.data(), displacements.data(),
                                             convert_to_mpi_datatype<T>(),
                                             grid().comms()[1]);
                    assert(err == MPI_SUCCESS);
                }

                {
                    Tracer tracer("scatter in stage 2 in replicate_in_all");
                    auto C = allresult.data(); int ldc = allresult.ld();
                    auto A = recvbuf.data(); int lda = recvbuf.ld();
                    for (int pi = 0; pi < grid().dims()[1]; pi++) {
                        int dpi = displacements[pi];
                        auto j12 = global_to_local_index({j1, j2}, pi, grid().dims()[1], 0);
                        int nn = j12[1] - j12[0];
                        for (int j = 0; j < nn; j++) {
                            int jmm = j*mm;
                            int gj = (j + j12[0]) * grid().dims()[1] + pi;
                            for (int i = 0; i < mm; i++) {
//                                allresult(i, gj - j1) = recvbuf.data()[dpi + i + jmm];
                                C[i+(gj-j1)*ldc] = A[dpi+i+jmm];
                            }
                        }
                    }
                }
            }
            return allresult;
        }
        LMatrix<T> replicate_in_all_1stage(int i1, int j1, int i2, int j2) {
            Tracer tracer(__FUNCTION__ );
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
            int mm = i12[1] - i12[0];
            int nn = j12[1] - j12[0];
            int np = grid().dims()[0], nq = grid().dims()[1];
            int max_mm = (i2-i1 + np-1) / np;
            int max_nn = (j2-j1 + nq-1) / nq;
            assert(mm<=max_mm && nn <=max_nn);

            LMatrix<T> recvbuf(max_mm, max_nn*grid().np());
            LMatrix<T> result(i2-i1, j2-j1);
            LMatrix<T> sendbuf(max_mm, max_nn);
            copy_block(mm, nn, &data()[i12[0] + j12[0] * lld()], lld(), sendbuf.data(), sendbuf.ld());
            auto block_sizes = list_local_part_sizes(i1, i2, grid().dims()[0]);
            {
                Tracer tracer("Allgather in stage 1");
                auto mpi_type = convert_to_mpi_datatype<T>();
                int err = MPI_Allgather(sendbuf.data(), max_mm*max_nn, mpi_type, recvbuf.data(), max_mm*max_nn, mpi_type, grid().comm());
                assert(err == MPI_SUCCESS);
            }
            {

                Tracer tracer("scatter in stage 1 in replicate_in_all");
                for (int pi = 0; pi < grid().np(); pi++) {
                    auto C = result.data(), ldc = result.ld(), A = &recvbuf.data()[max_mm*max_nn*pi];
                    auto pq = grid().convert_to_pq(pi);
                    auto i12 = global_to_local_index({i1, i2}, pq[0], grid().dims()[0], 0);
                    auto j12 = global_to_local_index({j1, j2}, pq[1], grid().dims()[1], 0 );
                    int mm = i12[1] - i12[0];
                    int nn = j12[1] - j12[0];
                    int np = grid().dims()[0];
                    int nq = grid().dims()[1];
                    for (int j = 0; j < nn; j++) {
                        int ldrecv = recvbuf.ld();
                        for (int i = 0; i < mm; i++) {
                            int gi = (i + i12[0]) * np + pq[0];
                            int gj = (j + j12[0]) * nq + pq[1];
//                                result(gi - i1, j) = recvbuf.data()[dpi + i + jmm];
                            C[gi-i1+(gj-j1)*ldc] = A[i+j*ldrecv];
                        }
                    }
                }
            }


            return result;
        }

        // row dim (dim[0]) are no longer distributed; instead it's replicated on all row processs
        LMatrix<T> replicate_in_columns(int i1, int j1, int i2, int j2) {
            Tracer tracer(__FUNCTION__ );
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
            int mm = i12[1] - i12[0];
            int nn = j12[1] - j12[0];

            LMatrix<T> sendbuf(mm, nn);
            copy_block(mm, nn, &data()[i12[0]+j12[0]*lld()], lld(), sendbuf.data(), sendbuf.ld());
            auto block_sizes = list_local_part_sizes(i1, i2, grid().dims()[0]);
            LMatrix<T> recvbuf(i2-i1, nn);
            std::vector<int> displacements( grid().dims()[0], 0 );
            std::vector<int> recv_counts( grid().dims()[0], 0);

            for(int i=1; i<displacements.size(); i++) {
                displacements[i] = displacements[i-1] + block_sizes[i-1]*nn;
                recv_counts[i] = block_sizes[i]*nn;
            }
            recv_counts[0] = block_sizes[0]*nn;
#ifdef DEBUG_REPLICATE
            if(grid().ranks()[0]==0) {
                fmt::print("P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1],recv_counts, displacements );
            } else {
//                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
            }
#endif
            {
                Tracer tracer("Allgatherv in replicate_in_columns");
                int err = MPI_Allgatherv(sendbuf.data(), mm * nn, convert_to_mpi_datatype<T>(),
                                         recvbuf.data(), recv_counts.data(), displacements.data(),
                                         convert_to_mpi_datatype<T>(),
                                         grid().comms()[0]);
                assert(err == MPI_SUCCESS);
            }
            LMatrix<T> result(i2-i1, nn);
            {
                Tracer tracer("scatter in replicate_in_columns");
                auto C = result.data();
                auto A = recvbuf.data();
                int np = grid().dims()[0];
                int ldresult = result.ld();
                for (int pi = 0; pi < grid().dims()[0]; pi++) {
                    auto i12 = global_to_local_index({i1, i2}, pi, np, 0);
                    int mm = i12[1] - i12[0];
                    int dpi = displacements[pi];

                    for (int j = 0; j < nn; j++) {
                        int jmm = j*mm;
                        int nppi = i12[0]*np + pi;
                        for (int i = 0; i < mm; i++) {
                            int gi = (i + i12[0]) * np + pi;
                            //result(gi - i1, j) = recvbuf.data()[dpi + i + jmm];
                            C[i*np+nppi-i1 + j*ldresult] = A[dpi + i + jmm];
                        }
                    }
                }
            }
            return result;
        }

        LMatrix<T> transpose_and_replicate_in_rows(int i1, int j1, int i2, int j2) {
            Tracer tracer(__FUNCTION__ );
            int p = grid().ranks()[0], q = grid().ranks()[1];
            int np = grid().dims()[0], nq = grid().dims()[1];
            assert(np == nq);
            auto i12 = global_to_local_index({i1, i2}, p, np, 0);
            auto j12 = global_to_local_index({j1, j2}, q, nq, 0);
            int mm = i12[1] - i12[0], nn = j12[1] - j12[0];

            auto ri12 = global_to_local_index({j1, j2}, p, np, 0);
            auto rj12 = global_to_local_index({i1, i2}, q, nq, 0);
            int rmm = ri12[1] - ri12[0], rnn = rj12[1] - rj12[0];

            std::vector<T> buf(rmm*rnn);
            int ldbuf = rmm;

            if(p == q) {
                assert(ldbuf == nn);
                copy_transposed_block(mm, nn, &m_data[i12[0]+j12[0]*lld()], lld(), buf.data(), ldbuf);
            } else {
                std::vector<T> sendbuf(mm*nn);
                copy_transposed_block(mm, nn, &m_data[i12[0]+j12[0]*lld()], lld(), sendbuf.data(), nn);
                MPI_Status status;
                int err =
                MPI_Sendrecv(sendbuf.data(), mm*nn, convert_to_mpi_datatype<T>(), m_grid.convert_to_linear(q,p), p+q,
                             buf.data(), rmm*rnn, convert_to_mpi_datatype<T>(), m_grid.convert_to_linear(q,p), p+q,
                             m_grid.comm(), &status);
                assert(err == MPI_SUCCESS);
            }
            DMatrix<T> At(grid(), dims()[1], dims()[0],buf.data(), ldbuf, j1, i1);
            return At.replicate_in_rows(j1, i1, j2, i2);
        }

        // replicate in rows means the processes in the same row
        // will return the same local matrix.
        LMatrix<T> replicate_in_rows(int i1, int j1, int i2, int j2) {
            Tracer tracer(__FUNCTION__ );
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
            int mm = i12[1] - i12[0];
            int nn = j12[1] - j12[0];

            LMatrix<T> sendbuf(mm, nn);
            copy_block(mm, nn, &data()[i12[0]+j12[0]*lld()], lld(), sendbuf.data(), sendbuf.ld());
            auto block_sizes = list_local_part_sizes(j1, j2, grid().dims()[1]);
            LMatrix<T> recvbuf(mm, j2-j1);
            std::vector<int> displacements( grid().dims()[1], 0 );
            std::vector<int> recv_counts( grid().dims()[1], 0);

            for(int i=1; i<displacements.size(); i++) {
                displacements[i] = displacements[i-1] + block_sizes[i-1]*mm;
                recv_counts[i] = block_sizes[i]*mm;
            }
            recv_counts[0] = block_sizes[0]*mm;
#ifdef DEBUG_REPLICATE
            if(grid().ranks()[1]==0) {
                fmt::print("P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1],recv_counts, displacements );
            } else {
//                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
            }
#endif
            {
                Tracer tracer("Allgatherv in replicate_in_rows");
                int err = MPI_Allgatherv(sendbuf.data(), mm * nn, convert_to_mpi_datatype<T>(),
                                         recvbuf.data(), recv_counts.data(), displacements.data(),
                                         convert_to_mpi_datatype<T>(),
                                         grid().comms()[1]);
                assert(err == MPI_SUCCESS);
            }
            LMatrix<T> result(mm,j2-j1);

            {
                Tracer tracer("scatter in replicate_in_rows");
                for (int pi = 0; pi < grid().dims()[1]; pi++) {
                    auto j12 = global_to_local_index({j1, j2}, pi, grid().dims()[1], 0);
                    int nn = j12[1] - j12[0];
                    for (int j = 0; j < nn; j++) {
                        int gj = (j + j12[0]) * grid().dims()[1] + pi;
                        for (int i = 0; i < mm; i++) {
//                        int gi = (i+i12[0])*grid().dims()[0] + pi;
                            result(i, gj - j1) = recvbuf.data()[displacements[pi] + i + j * mm];
                        }
                    }
                }
            }
            return result;
        }

        void dereplicate_in_columns(LMatrix<T> A, int i1, int j1, int i2, int j2) {
//            Tracer tracer(__FUNCTION__ );
            int mm = A.dims()[0], nn = A.dims()[1];
            assert(mm == i2-i1);
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
            int np = grid().dims()[0], nq = grid().dims()[1];
            int p = grid().ranks()[0];
            for(int j=0; j<nn; j++) {
                for(int i=0; i<i12[1]-i12[0]; i++) {
//                    (*this)( i12[0]+i,  (j12[0]+j) ) = A( (i+i12[0])*np+p - i1, j);
                    data()[i12[0]+i + (j12[0]+j)*lld()] = A( (i+i12[0])*np+p - i1, j);
                }
            }
        }

        void dereplicate_in_rows(LMatrix<T> A, int i1, int j1, int i2, int j2) {
            Tracer tracer(__FUNCTION__ );
            int mm = A.dims()[0], nn = A.dims()[1];
            assert(nn == j2-j1);
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
            int  nq = grid().dims()[1];
            int q = grid().ranks()[1];
            for(int j=0; j<j12[1]-j12[0]; j++) {
                for(int i=0; i<mm; i++) {
                    data()[i12[0]+i + (j12[0]+j)*lld()] = A( i, (j+j12[0])*nq+q-j1);
                }
            }
        }

        void dereplicate_in_all(LMatrix<T> A, int i1, int j1, int i2, int j2) {
            Tracer tracer(__FUNCTION__ );
            int mm = A.dims()[0], nn = A.dims()[1];
            // printf("mm::%d [%d,%d] nn::%d [%d, %d]\n", mm, i1, i2, nn, j1, j2);
            assert(mm == i2-i1 && nn == j2-j1);
            auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
            auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
            int np = grid().dims()[0], nq = grid().dims()[1];
            int p = grid().ranks()[0], q = grid().ranks()[1];
            for(int j=0; j<j12[1]-j12[0]; j++) {
                for(int i=0; i<i12[1]-i12[0]; i++) {
                    data()[i12[0]+i + (j12[0]+j)*lld()] = A( (i+i12[0])*np+p - i1, (j+j12[0])*nq+q-j1);
                }
            }
        }

        // align: the process that element 0 is at;
        static inline std::array<int,2> global_to_local_index(std::array<int,2> ij, int p, int np, int align) {
//            printf("ij={%d,%d}, p=%d, np=%d, align=%d\n", ij[0], ij[1], p, np, align);
            int gi = ij[0], gj = ij[1]; 
            int li = ceil(1.0*(gi-(p-align+np)%np)/np);
            int lj = floor(1.0*(gj-(p-align+np)%np+np-1)/np);
            return {li, lj};
        }

        inline std::vector<int> list_local_part_sizes(int i1, int i2,  int np) {
            std::vector<int> l(np, 0);
            for(int i=0; i<np; i++) {
                auto ii = global_to_local_index({i1,i2}, i, np, 0);
                l[i] = ii[1]-ii[0];
            }
            return l;
        }


        LMatrix<T> copy_upper_to_lower(LMatrix<T> A, int j1, int j2) {
            Tracer tracer(__FUNCTION__ );
            assert(grid().dims()[0] == grid().dims()[1]);
            int p = grid().ranks()[0], q = grid().ranks()[1];
            auto lij = global_to_local_index({j1,j2}, p, grid().dims()[0], 0);
            int mm = A.dims()[0];
            int recv_nn = lij[1] - lij[0];
            LMatrix<T> recvbuf(mm, recv_nn);
            MPI_Status status;
            int err = MPI_Sendrecv(A.data(),mm*A.dims()[1], convert_to_mpi_datatype<T>(), grid().convert_to_linear(q,p), q+p,
                         recvbuf.data(), mm*recv_nn, convert_to_mpi_datatype<T>(), grid().convert_to_linear(q,p), q+p,
                         grid().comm(), &status);
            assert(err == MPI_SUCCESS);

            return recvbuf;
        }

        LMatrix<T> copy_lower_to_upper(LMatrix<T> A, int i1, int i2) {
            Tracer tracer(__FUNCTION__ );
            assert(grid().dims()[0] == grid().dims()[1]);
            int p = grid().ranks()[0], q = grid().ranks()[1];
            auto lij = global_to_local_index({i1,i2}, p, grid().dims()[0], 0);
            int nn = A.dims()[1];
            int recv_mm = lij[1] - lij[0];
            LMatrix<T> recvbuf(nn, recv_mm);
            MPI_Status status;
            int err = MPI_Sendrecv(A.data(),nn*A.dims()[0], convert_to_mpi_datatype<T>(), grid().convert_to_linear(q,p), q+p,
                         recvbuf.data(), nn*recv_mm, convert_to_mpi_datatype<T>(), grid().convert_to_linear(q,p), q+p,
                         grid().comm(), &status);
            assert(err == MPI_SUCCESS);
            return recvbuf;
        }


        void print_in_process_sequence(LMatrix<T> A, const std::string &title) {
            for(int i=0; i<dims()[0]; i++) {
                for(int j=0; j<dims()[1]; j++) {
                    if(grid().ranks()[0] == i && grid().ranks()[1] == j) {
                        A.print(fmt::format("P[{},{}]: {}", grid().ranks()[0], grid().ranks()[1], title));
                    }
                    grid().barrier();
                }
            }
        }

        void permute_rows_inverse_ipiv(int i1, int j1, int i2, int j2, int* ipiv, int n_ipiv) {
            Tracer tracer(__FUNCTION__);
            for (int i = i1; i < i2; i++)
                m_permutation[i] = i;
            for (int i = i1; i < i1 + n_ipiv; i++) {
                std::swap(m_permutation[i],
                          m_permutation[ipiv[i - i1] - 1 + i1]);  // ipiv uses Fortran 1-base array index.
            }
            auto inv_perm = m_permutation;
            for(int i=0; i<n_ipiv; i++) {
                inv_perm[m_permutation[i]] = i;
            }
            if(grid().rank()==0) {
//                fmt::print("perm={}\n", m_permutation);
//                fmt::print("inverse perm={}\n", inv_perm);
            }
            grid().barrier();
            permute_rows_perm(i1, j1, i2, j2, inv_perm.data(), n_ipiv);

        }

        void permute_rows_ipiv(int i1, int j1, int i2, int j2, int* ipiv, int n_ipiv) {
            Tracer tracer(__FUNCTION__);
            for (int i = i1; i < i2; i++)
                m_permutation[i] = i;
            for (int i = i1; i < i1 + n_ipiv; i++) {
//                fmt::print("i={}, ipiv[i-i1]-1+i1={}\n", i, ipiv[i-i1]-1+i1);
                std::swap(m_permutation[i],
                          m_permutation[ipiv[i - i1] - 1 + i1]);  // ipiv uses Fortran 1-base array index.
            }
            permute_rows_perm(i1, j1, i2, j2, m_permutation.data(), n_ipiv);

        }
//            fmt::print("ipiv={}, i1,i2={},{}\nm_permutations={}\n", ipiv, i1, i2, std::vector<int>(m_permutation.begin()+i1, m_permutation.begin()+i2));
        void permute_rows_perm(int i1, int j1, int i2, int j2, int* perm, int n_perm){
            auto j12 = global_to_local_index({j1, j2}, grid().ranks()[1], grid().dims()[1], 0);
            auto i12 = global_to_local_index({i1, i2}, grid().ranks()[0], grid().dims()[0], 0);
            int nn = j12[1] - j12[0], mm = i12[1] - i12[0];
            int p = grid().ranks()[0], np = grid().dims()[0];
            std::vector<int> recv_process_list, send_process_list;

            // local rows.
            struct PermInfo {
                int process;
                int local_index;
            };
//            if(grid().ranks()[1]==0){
//            if (grid().ranks()[0] == 0) fmt::print("nn={}, ipiv={}, m_perm={}\n", nn, ipiv, m_permutation);
            std::vector<int> send_counts(np, 0), recv_counts(np, 0);
//            fmt::print("Process {}\n", grid().ranks()[0]);
            std::vector<PermInfo> recv_info;
            for (int i = i12[0]; i < i12[1]; i++) {
                int gi = i * np + p;
                int recv = perm[gi];
                if (recv != gi) {
//                    fmt::print("li={} gi={} needs to recv {} from process {}\n", i, gi, recv, recv % np);
                    recv_counts[recv % np]++;
                    recv_info.push_back({recv % np, i});
                }
            }
//            fmt::print("recv_counts={}\n", recv_counts);
            std::vector<PermInfo> send_info;
            for (int i = i1; i < i2; i++) {
                if (perm[i] % np == p) {
                    if (i != perm[i]) {
//                        fmt::print("need to send li={},gi={} to process {} at gi={}\n",(m_permutation[i] - p) / np, m_permutation[i], i % np, i);
                        send_counts[i % np]++;
                        send_info.push_back({i % np, (perm[i] - p) / np});
                    }
                }
            }
//            fmt::print("send_coutns={}\n", send_counts);
            auto comp_by_first = [](PermInfo a, PermInfo b) { return a.process < b.process; };
            std::stable_sort(send_info.begin(), send_info.end(), comp_by_first);
            std::stable_sort(recv_info.begin(), recv_info.end(), comp_by_first);
//                    fmt::print("sorted\nsend_info={}\nrecv_info={}\n", send_info, recv_info);

            int send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0);
            std::vector<int> send_counts_scan(np);
            std::exclusive_scan(send_counts.begin(), send_counts.end(), send_counts_scan.begin(), 0);
//            fmt::print("send_counts_scan={}\n", send_counts_scan);
            LMatrix<T> sendbuf(send_count * nn, 1);
            T *sbuf = sendbuf.data();
            {
                Tracer tracer("copy to sendbuf in permute rows");
                int ldm = lld();
                for (int ii = 0; ii < send_info.size(); ii++){
                    int offset = ldm * j12[0] + send_info[ii].local_index;
#pragma GCC ivdep
                    for (int i = 0; i < nn; i++) {
//                        auto li = send_info[ii].local_index;
                        sbuf[i + nn * ii] = m_data[offset + i * ldm];
                    }

//                  for (int i = 0; i < nn; i++) sbuf[i + nn * ii] = m_data[i];
                }
            }

            for (int i = 0; i < send_counts.size(); i++) {
                send_counts[i] *= nn;
                send_counts_scan[i] *= nn;
            }

            int recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
            std::vector<int> recv_counts_scan(np);
            std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_counts_scan.begin(), 0);
//            fmt::print("recv_counts_scan={}\n", recv_counts_scan);
            LMatrix<T> recvbuf(recv_count * nn, 1);
//                    T* rbuf = recvbuf.data();

            for (int i = 0; i < recv_counts.size(); i++) {
                recv_counts[i] *= nn;
                recv_counts_scan[i] *= nn;
            }
            {
                Tracer tracer("Alltoallv in permute rows");
                MPI_Alltoallv(sendbuf.data(), send_counts.data(), send_counts_scan.data(), convert_to_mpi_datatype<T>(),
                              recvbuf.data(), recv_counts.data(), recv_counts_scan.data(), convert_to_mpi_datatype<T>(),
                              grid().comms()[0]);
            }

            T *rbuf = recvbuf.data();
            {
                Tracer tracer("copy from recvbuf in permute rows");
                for (int ii = 0; ii < recv_info.size(); ii++) {
                    auto li = recv_info[ii].local_index;
#pragma GCC ivdep
                    for (int i = 0; i < nn; i++)
                        m_data[li + (i + j12[0]) * lld()] = rbuf[i + nn * ii];
                }
            }


        }

        void swap_rows(int row1, int row2, int j1, int j2) {
            int p = grid().ranks()[0], q = grid().ranks()[1], np = grid().dims()[0], nq = grid().dims()[1];
            auto j12 = global_to_local_index({j1, j2}, q, nq, 0); // must swap the whole row;

            if(row1 % np == row2 % np && row1 % np == p) { // local swap rows
                int l1 = (row1 - p) / np, l2 = (row2 - p) / np;  // local row inices
                for(int jj = j12[0]; jj < j12[1]; jj++) {
                    std::swap(data()[l1 + jj * lld()], data()[l2 + jj * lld()]);
                }
            } else { // cross process swap rows
                int nn = j12[1] - j12[0];
                std::vector<T> recvbuf(nn);
                std::vector<T> sendbuf(nn);
                MPI_Status status;
//                if (q==0) fmt::print("P{} and P{} are swapping row {} and {}\n", row1 % np, row2 % np, row1, row2);
                if (p == row1 % np) {
                    int l1 = (row1 - p) / np;
                    for(int jj=0; jj<nn; jj++) {
                        sendbuf[jj] = data()[l1 + (j12[0] + jj) * lld()];
                    }
//                        fmt::print("P{} sending to P{} and recv'ing from P{}\n", p, row2 % np, row2 % np);
                    MPI_Sendrecv(sendbuf.data(), nn,
                                 convert_to_mpi_datatype<T>(), row2 % np, row1,
                                 recvbuf.data(), nn, convert_to_mpi_datatype<T>(), row2 % np,
                                 row2, grid().comms()[0], &status);
                    for(int jj=0; jj<nn; jj++) {
                        data()[l1 + (j12[0] + jj) * lld()] = recvbuf[jj];
                    }
                } else if (p == row2 % np) {
                    int l2 = (row2 - p) / np;
                    for(int jj=0; jj<nn; jj++) {
                        sendbuf[jj] = data()[l2 + (j12[0] + jj) * lld()];
                    }
//                        fmt::print("P{} sending to P{} and recv'ing from P{}\n", p, row1 % np, row1 % np);
                    MPI_Sendrecv(sendbuf.data(), nn,
                                 convert_to_mpi_datatype<T>(), row1 % np, row2,
                                 recvbuf.data(), nn, convert_to_mpi_datatype<T>(), row1 % np, row1,
                                 grid().comms()[0], &status);
                    for(int jj=0; jj<recvbuf.size(); jj++) {
                        data()[l2+ (j12[0] + jj) * lld()] = recvbuf[jj];
                    }
                }


            }
        }

        void substract(DMatrix<T>& A) {
            auto i12 = global_to_local_index({0, A.dims()[0]}, A.grid().ranks()[0], A.grid().dims()[0], 0);
            auto j12 = global_to_local_index({0, A.dims()[1]}, A.grid().ranks()[1], A.grid().dims()[1], 0);
            for(int jj=j12[0]; jj<j12[1]; jj++) {
                for(int ii=i12[0]; ii<i12[1]; ii++) {
                    m_data[ii+jj*lld()] -= A.data()[ii+jj*A.lld()];
                }
            }
        }




    };


    template<typename T> size_t LMatrix<T>::total_allocated_bytes {0};
    template<typename T> size_t LMatrix<T>::maximum_allocated_bytes {0};

    template<typename T>
    struct DMatrixDescriptor {
        DMatrix<T> matrix;
        int i1, j1, i2, j2;
        int m() {
            return i2 - i1;
        }
        int n() {
            return j2 - j1;
        }
        LMatrix<T> replicate_in_rows() {
            return matrix.replicate_in_rows(i1, j1, i2, j2);
        }
        LMatrix<T> replicate_in_columns() {
            return matrix.replicate_in_columns(i1, j1, i2, j2);
        }
        LMatrix<T> replicate_in_all() {
            return matrix.replicate_in_all(i1, j1, i2, j2);
        }
        void print(const std::string &title) {
            auto A = replicate_in_all();
            if(matrix.grid().rank()==0) A.print(title);
        }
    };

}








// #pragma once

// #include <mpi.h>
// #include <cassert>
// #include <random>
// #include <cstdio>
// #include <chrono>
// #include <functional>
// #include <memory>
// #include <iostream>
// #include <chrono>
// #define FMT_HEADER_ONLY 1
// #include <fmt/format.h>
// #include <fmt/ranges.h>
// #include "utils.h"

// #define ASSERT_NOT_IMPLEMENTED assert(0)

// namespace DArray {
//     template<typename T>
//     void copy_block(int m, int n, T* src, int ldsrc, T* dest, int lddest) {
// //            Tracer tracer("copy_block");

//         for(int j=0; j<n; j++) {
//             int lj1 = lddest*j, lj2 = ldsrc*j;
//             for(int i=0; i<m; i++) {
//                 dest[i+lj1] = src[i+lj2];
//             }
//         }
//     }

//     // dest = src^T
//     // src: m*n.
//     // dest: n*m
//     template<typename T>
//     void copy_transposed_block(int m, int n, T* src, int ldsrc, T* dest, int lddest) {
//         for(int j=0; j<n; j++) {
//             int lj1 = lddest*j, lj2 = ldsrc*j;
//             for(int i=0; i<m; i++) {
//                 dest[j+i*lddest] = src[i+lj2];
//             }
//         }
//     }


//     template<typename T>
//     class LMatrix;
//     template<typename T>
//     class DMatrix;
//     class Grid;

//     enum Layout { RowMajor, ColMajor };

//     template<typename T>
//     MPI_Datatype convert_to_mpi_datatype() {
//         return MPI_BYTE;
//     }

//     template<>
//     MPI_Datatype convert_to_mpi_datatype<float>() {
//         return MPI_FLOAT;
//     }

//     inline int size_of_local_part(int globalsize, int p, int np) {
//         return (globalsize - p + np -1 ) / np;
//     }

//     //
//     inline std::vector<int> list_local_part_sizes(int n, int np) {
//         std::vector<int> l(np, 0);
//         for(int i=0; i<np; i++) {
//             l[i] = size_of_local_part(n, i, np);
//         }
//         return l;
//     }

//     inline std::vector<int> scan_local_part(int n, int np) {
//         std::vector<int> l(np, 0);
//         for(int i=1; i<np; i++) {
//             l[i] = size_of_local_part(n, i - 1, np) + l[i - 1];
//         }
//         return l;
//     }




//     class Grid {
//         std::array<int,2> m_dims;
//         std::array<int,2> m_ranks;
//         std::array<MPI_Comm,2> m_comms; // row comm, col comm
//         MPI_Comm m_comm;
//         int m_np;
//         int m_rank;
//         Layout m_layout;
//     public:
//         // given global comm, create p*q grid.
//         Grid(const MPI_Comm &comm, int p, int q) : m_comm{comm}, m_dims{p, q}, m_layout{ColMajor} {
//             MPI_Comm_size(comm, &m_np);
//             assert(p != 0 && q != 0 && p * q <= m_np);

//             MPI_Comm_rank(comm, &m_rank);
//             if (m_layout == ColMajor) {
//                 m_ranks[0] = m_rank % p;
//                 m_ranks[1] = m_rank / p;
//                 int myrank;
//                 auto err1 = MPI_Comm_split(comm, m_rank / p, m_rank % p, &m_comms[0]); // col comm
//                 assert(err1 == MPI_SUCCESS);
//                 MPI_Comm_rank(m_comms[0], &myrank);
//                 assert(myrank == m_ranks[0]);
//                 auto err2 = MPI_Comm_split(comm, m_rank % p, m_rank / p, &m_comms[1]); // row comm
//                 assert(err2 == MPI_SUCCESS);
//                 MPI_Comm_rank(m_comms[1], &myrank);
//                 assert(myrank == m_ranks[1]);
//             }
//         }

//         int rank() { return m_rank; }

//         std::array<int,2> ranks() { return m_ranks; }

//         int np() { return m_np; }

//         std::array<int,2>  dims() { return m_dims; }

//         MPI_Comm comm() { return m_comm; }

//         std::array<MPI_Comm,2> comms() { return m_comms; }

//         Layout layout() { return m_layout; }

//         int convert_to_linear(int i, int j) {
//             if( m_layout == ColMajor) {
//                 return i+j*m_dims[0];
//             } else {
//                 ASSERT_NOT_IMPLEMENTED;
// 		        return 0;
//             }
//         }
//         std::array<int,2> convert_to_pq(int p) {
//             if(layout() == ColMajor) {
//                 int np = dims()[0];
//                 return {p%np, p/np};
//             } else {
//                 ASSERT_NOT_IMPLEMENTED;
//                 return {};
//             }
//         }

//         void barrier() {
//             MPI_Barrier(comm());
//         }

//         void barrier_col() {
//             MPI_Barrier(comms()[0]);
//         }

//         void barrier_row() {
//             MPI_Barrier(comms()[1]);
//         }

//         static void np_to_pq(int np, int &p, int &q) {
//             int k = round(sqrt(np));
//             for(; k>0; k--) {
//                 if (np%k==0) {
//                     p = k;
//                     q = np/k;
//                     break;
//                 }
//             }
//         }

//         void for_each_process_in_column(std::function<void(void)> func) {
//             for(int pi = 0; pi<dims()[0]; pi++) {
//                 if(ranks()[0] == pi) {
//                     func();
//                 }
//                 barrier_col();
//             }
//         }

//         void for_each_process_in_row(std::function<void(void)> func) {
//             for(int qi = 0; qi<dims()[1]; qi++) {
//                 if(ranks()[1] == qi) {
//                     func();
//                 }
//                 barrier_row();
//             }
//         }

//         void for_each_process_in_grid(std::function<void(void)> func) {
//             barrier();
//             for(int pi = 0; pi<dims()[0]; pi++) {
//                 for(int pj = 0; pj<dims()[1]; pj++) {
//                     if (ranks()[0] == pi && ranks()[1] == pj) {
//                         func();
//                     }
//                     barrier();
//                 }
//             }
//             barrier();
//         }
//         template<typename T>
//         void print_row_replicated(LMatrix<T> A, const std::string &title) {
//             barrier();
//             if(ranks()[1] == 0) {
//                 for (int pi = 0; pi < dims()[0]; pi++) {
//                     if (ranks()[0] == pi) {
//                         A.print(fmt::format("P[{},*] {}", pi, title));
//                     }
//                     barrier_col();
//                 }
//             }
//             barrier();
//         }
//         template<typename T>
//         void print_column_replicated(LMatrix<T> A, const std::string &title) {
//             barrier();
//             if(ranks()[0] == 0) {
//                 for (int pi = 0; pi < dims()[1]; pi++) {
//                     if (ranks()[1] == pi) {
//                         A.print(fmt::format("P[*,{}] {}", pi, title));
//                     }
//                     barrier_row();
//                 }
//             }
//             barrier();
//         }
//         template<typename T>
//         void print(LMatrix<T> A, const std::string &title) {
//             barrier();
//             for(int pi = 0; pi<dims()[0]; pi++) {
//                 for(int pj = 0; pj<dims()[1]; pj++) {
//                     if (ranks()[0] == pi && ranks()[1] == pj) {
//                         A.print(fmt::format("P[{},{}] {}", pi,pj,  title));
//                     }
//                     barrier();
//                 }
//             }
//             barrier();
//         }
//         //



//     };

//     inline std::ostream &operator<<(std::ostream &os, Grid &grid) {
//         if (grid.rank() == 0)
//             return os << "Grid: np=" << grid.np() << " layout " << grid.layout() << std::endl;
//         else {
//             return os << "ranks: [" << grid.ranks()[0] << ", " << grid.ranks()[1] << "]" << std::endl;
//         }
//     }

//     template<typename T>
//     class LMatrix { // local matrix
//         static size_t total_allocated_bytes;
//         static size_t maximum_allocated_bytes;
//         std::shared_ptr<T[]> m_storage;
//         T* m_data;
//         std::array<int,2> m_dims;
//         Layout layout;
//         int m_ld; // leading dimension.
//     public:
//         LMatrix() : m_dims{0, 0}, layout{ColMajor}, m_ld{0}, m_data{} {}
//         LMatrix(int m, int n) : m_dims{m, n}, layout{ColMajor} {
// //            fmt::print("new {}\n", m*n);
//             if(m*n<0) {
//                 fmt::print(Backtrace());
//             }
//             std::shared_ptr<T[]> s(new T[m*n]);
//             m_storage = s;
//             m_ld = m;
//             m_data = m_storage.get();


//         }
//         LMatrix(int m, int n, int ld) : m_dims{m, n}, layout{ColMajor} {
//             std::shared_ptr<T[]> s(new T[ld*n]);
//             m_storage = s;
//             m_data = m_storage.get();
//             m_ld = ld;


//         }
//         LMatrix(int m, int n, int ld, T* buffer)
//             : m_dims{m, n}, layout{ColMajor},  m_ld{ld}, m_data{buffer}, m_storage{nullptr} {
//         }

// //        ~LMatrix() {
// //
// //        }

//         T &operator[](int i) {return m_data[i];}
//         T operator[](int i) const {return m_data[i];}
//         T &operator()(int i,int j) {return m_data[i+j*m_ld];}
//         T operator()(int i, int j) const {return m_data[i+j*m_ld];}
//         int& ld() {return m_ld; }
//         void set_ld(int ld_) {m_ld = ld_; }
//         T* data() {return m_data;}
//         std::array<int,2> dims() {return m_dims;}

//         void print(const std::string title) {
//             std::cout << title << std::endl;
//             for(int i=0; i < m_dims[0]; i++) {
//                 for(int j=0; j < m_dims[1]; j++) {
//                     printf("%8.3f ", (*this)(i,j));
//                 }
//                 printf("\n");
//             }
//         }
//         void save_to_file(const std::string filepath) {
//             FILE *f = fopen(filepath.c_str(), "w");
//             assert(f);
//             for(int i=0; i < m_dims[0]; i++) {
//                 for(int j=0; j < m_dims[1]; j++) {
//                     fmt::print(f, "{} ", (*this)(i,j));
//                 }
//                 fmt::print(f, "\n");
//             }
//             fclose(f);
//         }
//         void set_constant(T a) {
//             for (int i = 0; i < m_dims[0]; i++) {
//                 for (int j = 0; j < m_dims[1]; j++) {
//                     (*this)(i,j) = a;
//                 }
//             }
//         }

//         void set_normal_seed(T mean, T stddev, int seed) {
//             std::mt19937 generator(seed);
//             std::normal_distribution<T> distribution(mean, stddev);
//             for (int i = 0; i < m_dims[0]; i++) {
//                 for (int j = 0; j < m_dims[1]; j++) {
//                     m_data[i+j*m_ld] = distribution(generator);
//                 }
//             }
//         }

//         void set_function(std::function<T (int, int)> func) {
//             for (int i = 0; i < dims()[0]; i++) {
//                 for (int j = 0; j < dims()[1]; j++) {
//                     (*this)(i,j) = func(i, j);
//                 }
//             }
//         }

//         LMatrix<T> transpose() {
//             int m = m_dims[0], n = m_dims[1];
//             LMatrix<T> At(n, m);
//             copy_transposed_block(m, n, m_data, m_ld, At.data(), At.ld());
//             return At;
//         }

//         void clear_upper() {
//             int m = m_dims[0], n = m_dims[1];
//             for(int j=0; j<n; j++) {
//                 for(int i=0; i<j; i++)
//                     m_data[i+j*m_ld] = 0;
//             }
//         }

//         void clear_lower() {
//             int m = m_dims[0], n = m_dims[1];
//             for(int j=0; j<n; j++) {
//                 for(int i=j+1; i<m; i++)
//                     m_data[i+j*m_ld] = 0;
//             }
//         }

//         void set_zeros() {
//             for (int i = 0; i < dims()[0]; i++) {
//                 for (int j = 0; j < dims()[1]; j++) {
//                     (*this)(i,j) = 0;
//                 }
//             }
//         }

//     };


//     template<typename T>
//     class DMatrix {
//         Grid m_grid;
//         std::array<int,2> m_dims;
//         std::array<int,2> m_ldims;

//         std::shared_ptr<T[]> m_storage;
// //        std::vector<T> m_storage;
//         T *m_data;
//         int m_lld; // local leading dimension
// //        Layout m_layout;
//         T unused;
//         std::vector<int> m_permutation;
//         std::vector<int> m_ipiv;
//     public:
//         // allocating and owns memory
//         DMatrix(Grid &grid, int m, int n)
//                 : m_grid{grid}, m_dims{m, n}
//                 , m_permutation{std::move(std::vector(m, -1))}
//                 , m_ipiv{std::move(std::vector<int>(m, -1))} {
//             auto r = grid.ranks();
//             auto s = grid.dims();
//             m_ldims[0] = size_of_local_part(m, r[0], s[0]);
//             m_ldims[1] = size_of_local_part(n, r[1], s[1]);
//             m_lld = m_ldims[0];
//             std::shared_ptr<T[]> ss(new T[m_ldims[0] * m_ldims[1]]);
//             m_storage = ss;
//             m_data = m_storage.get();
// //            fmt::print("DMatrix: allocating {}MB\n", 1.0*m_ldims[0] * m_ldims[1] * sizeof(T)/1.0e6 );
//         }

//         // attaching to a buffer. The buffer is incomplete.
//         DMatrix(Grid &grid, int m, int n, T* buf, int ld, int i1, int j1)
//             : m_grid{grid}, m_dims{m,n}
//             , m_permutation{}
//             , m_ipiv{}
//             , m_storage{}
//             , m_data {nullptr}
//             , m_ldims{0,0}
//             , m_lld{ld}{
//             auto i12 = global_to_local_index({i1, m}, grid.ranks()[0], grid.dims()[0], 0);
//             auto j12 = global_to_local_index({j1, n}, grid.ranks()[1], grid.dims()[1], 0);
//             m_data = buf - (i12[0]+j12[0]*ld);
//         }


//         DMatrix<T> clone() {
//             auto A(*this);
//             std::shared_ptr<T[]> ss(new T[A.m_ldims[0] * A.m_ldims[1]]);
//             memcpy(ss.get(), m_storage.get(), sizeof(T)*A.m_ldims[0] * A.m_ldims[1]);
//             A.m_storage = std::move(ss);
//             A.m_data = A.m_storage.get();
//             return A;
//         }

//         std::array<int, 2> dims() { return m_dims;}
//         std::array<int, 2> ldims() { return m_ldims;}
//         Grid& grid() {return m_grid; }
// //        T &operator()(int i, int j) {return m_data[i + j * m_lld];}
// //        T operator()(int i, int j) const {return m_data[i + j * m_lld];}
//         T* data() {return m_data; }
//         int lld() {return m_lld; }
//         std::vector<int>& ipiv() {return m_ipiv; }


//          //global row,col indexing into the matrix; only the possessing process operate.
//         inline T& at(int i, int j) {
//             int np = grid().dims()[0], nq = grid().dims()[1];
//             int p = i % np, q = j % nq;
//             if(p == grid().ranks()[0] && q == grid().ranks()[1]) {
//                 return m_data[(i-p)/np + m_lld * (j-q)/nq];
//             } else {
//                 return unused;
//             }
//         }

//         inline T operator()(int i, int j)  {
//             int np = grid().dims()[0], nq = grid().dims()[1];
//             int p = i % np, q = j % nq;
//             int root = grid().convert_to_linear(p,q);
//             T val;
//             if(grid().rank() == root)  {
//                 val = m_data[(i-p)/np + m_lld * (j-q)/nq];
//             }
//             MPI_Bcast(&val, 1, convert_to_mpi_datatype<T>(), root, grid().comm());
//             return val;
//         }


//         void set_uniform(T a, T b) {
//             std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
//             std::uniform_real_distribution<T> distribution(a, b);
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 for (int j = 0; j < m_ldims[1]; j++) {
//                     m_data[i+j*m_lld] = distribution(generator);
//                 }
//             }
//         }

//         void set_normal(T mean, T stddev) {
//             std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
//             std::normal_distribution<T> distribution(mean, stddev);
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 for (int j = 0; j < m_ldims[1]; j++) {
// //                    (*this)(i, j) = distribution(generator);
//                     m_data[i+j*m_lld] = distribution(generator);
//                 }
//             }
//         }

//         void set_normal_seed(T mean, T stddev, int seed) {
//             std::mt19937 generator(seed);
//             std::normal_distribution<T> distribution(mean, stddev);
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 for (int j = 0; j < m_ldims[1]; j++) {
//                     m_data[i+j*m_lld] = distribution(generator);
//                 }
//             }
//         }

//         void set_constant(T a) {
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 for (int j = 0; j < m_ldims[1]; j++) {
//                     //(*this)(i, j) = a;
//                     m_data[i+j*lld()] = a;
//                 }
//             }
//         }

//         void set_identity() {
//             auto d = m_grid.dims();
//             auto r = m_grid.ranks();
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 int gi = d[0] * i + r[0];
//                 for (int j = 0; j < m_ldims[1]; j++) {
//                     int gj = d[1] * j + r[1];
//                     if (gi==gj) (*this)(i, j) = 1;
//                     else (*this)(i,j) = 0;
//                 }
//             }
//         }

//         void set_zeros() {
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 for (int j = 0; j < m_ldims[1]; j++) {
//                     //  (*this)(i, j) = 0;
//                     m_data[i+j*lld()] = 0;
//                 }
//             }
//         }

//         void set_function(std::function<T (int, int)> func) {
//             auto d = m_grid.dims();
//             auto r = m_grid.ranks();
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 int gi = d[0] * i + r[0];
//                 for (int j = 0; j < m_ldims[1]; j++) {
//                     int gj = d[1] * j + r[1];
//                     // if(m_grid.rank() == 3) printf("gi::%d, gj::%d \n", gi, gj);
//                     m_data[i+j*m_lld] = func(gi,gj);
//                 }
//             }
//         }

//         void set_value_from_matrix(float* x, int ldx) {
//             auto d = m_grid.dims();
//             auto r = m_grid.ranks();
//             for (int i = 0; i < m_ldims[0]; i++) {
//                 int gi = d[0] * i + r[0];
//                 for (int j = 0; j < m_ldims[1]; j++) {
//                     int gj = d[1] * j + r[1];
//                     // if(m_grid.rank() == 3) printf("gi::%d, gj::%d \n", gi, gj);
//                     m_data[i+j*m_lld] = x[gi+gj*ldx];
//                 }
//             }
//         }

//         // forbenius norm
//         double fnorm() {
//             double sqr_sum = 0;
//             for(int i=0; i<ldims()[0]; i++) {
//                 for(int j=0; j<ldims()[1]; j++) {
//                     sqr_sum += m_data[i+j*m_lld] * m_data[i+j*m_lld];
//                 }
//             }
//             double  all_sum = 0;
//             MPI_Allreduce(&sqr_sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, grid().comm());
//             return sqrt(all_sum);
//         }

//         void print_by_process(const std::string &title)  {

//             auto d = m_grid.dims();
//             for(int pi=0; pi<grid().np(); pi++) {
//                 if(grid().rank() == pi){
//                     if (title != "") {
//                         printf("P[%3d,%3d]: ", m_grid.ranks()[0], m_grid.ranks()[1]);
//                         std::cout << title << std::endl;
//                     }
//                     auto r = m_grid.ranks();
//                     for (int i = 0; i < m_ldims[0]; i++) {
//                         for (int j = 0; j < m_ldims[1]; j++) {
//                             int gi = d[0] * i + r[0];
//                             int gj = d[1] * j + r[1];
//                             printf("[%2d,%2d] %8.3f  ", gi, gj, (*this)(i, j));
//                         }
//                         printf("\n");
//                     }
//                 }
//                 grid().barrier();
//             }
//         }

//         // destructive resize.
// //        void resize(int m, int n) {
// //            if (m==m_dims[0] && n==m_dims[1])
// //                return;
// //            m_dims[0] = m; m_dims[1] = n;
// //            auto r = m_grid.ranks();
// //            auto s = m_grid.dims();
// //            m_ldims[0] = (m - r[0] + s[0] - 1) / s[0];
// //            m_ldims[1] = (n - r[1] + s[1] - 1) / s[1];
// //            m_lld = m_ldims[0];
// //            delete[] m_data;
// //            m_data = new T[m_ldims[0] * m_ldims[1]];
// //        }


//         // collect DMatrix on a single process "root"; allocates memory.
//         LMatrix<T>  collect(int root) {
//             Tracer tracer(__FUNCTION__ );
//             if(root<0 || root>m_grid.np())
//                 assert(0);
//             int np = m_grid.dims()[0];
//             int nq = m_grid.dims()[1];
//             int m = m_dims[0], n = m_dims[1];

//             if (m_grid.rank() == root) {
//                 auto buffer = DArray::LMatrix<T>(m, n);
//                 int maxmm = (m+np-1)/np, maxnn = (n+nq-1)/nq;
//                 auto recvbuf = DArray::LMatrix<T>(maxmm, maxnn);
//                 for(int pi = 0; pi < np; pi++) {
//                     for(int qi = 0; qi < nq; qi++) {
//                         int mm = size_of_local_part(m, pi, np);
//                         int nn = size_of_local_part(n, qi, nq);
//                         recvbuf.set_ld(mm);
//                         int src = m_grid.convert_to_linear(pi, qi);
//                         if (src == root) {
//                             copy_block(mm, nn, data(), lld(), recvbuf.data(), recvbuf.ld());
//                         } else {
//                             MPI_Status status;
//                             int err = MPI_Recv(recvbuf.data(), mm * nn, convert_to_mpi_datatype<T>(), src, pi + qi, m_grid.comm(), &status);
//                             assert(err == MPI_SUCCESS);
//                         }
//                         scatter_local_to_global(mm, nn, recvbuf.data(), recvbuf.ld(), buffer.data(), buffer.ld(), pi, qi, np, nq);

//                     }
//                 }
//                 return buffer;
//             } else {
//                 if (lld() == ldims()[0]) {
//                     int err = MPI_Send(m_data, m_ldims[0] * m_ldims[1], convert_to_mpi_datatype<T>(), root,
//                                        m_grid.ranks()[0] + m_grid.ranks()[1], m_grid.comm());
//                     assert(err == MPI_SUCCESS);
//                     return LMatrix<T>();
//                 } else {
//                     ASSERT_NOT_IMPLEMENTED;
//                     return LMatrix<T>();
//                 }
//             }
//         }

//         //distribute accross all processes
//         void distribute(LMatrix<T> A, int i1, int j1, int i2, int j2, int root){
//             Tracer tracer(__FUNCTION__ );
//             if(root<0 || root>m_grid.np())
//                 assert(0);
//             int mm = A.dims()[0], nn = A.dims()[1];
//             assert(mm == i2-i1 && nn == j2-j1);
//             auto i12 = global_to_local_index({i1,i2}, m_grid.ranks()[0], m_grid.dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, m_grid.ranks()[1], m_grid.dims()[1], 0);
//             int np = m_grid.dims()[0], nq = m_grid.dims()[1];
//             int p = m_grid.ranks()[0], q = m_grid.ranks()[1];
//             printf("mm::%d, nn::%d, i12[0]::%d, i12[1]::%d, j12[0]::%d, j12[1]::%d, np::%d, nq::%d, p::%d, q::%d\n", mm, nn, i12[0], i12[1], j12[0], j12[1], np, nq, p, q);
//         }


//         void collect_and_print(const std::string &title, int root = 0) {
//             auto local_matrix = collect(root);
//             if(grid().rank()==root) {
//                 local_matrix.print(title);
//             }
//         }

//         void collect_and_print(const std::string &title, int i1, int j1, int i2, int j2) {
//             auto A = replicate_in_all(i1, j1, i2, j2);
//             if(grid().rank()==0) {
//                 A.print("title");
//             }
//         }

//         // scatter a contiguous buffer to global matrix
//         void scatter_local_to_global(int m, int n, T* src, int ldsrc, T* dest, int lddest, int p, int q, int np, int nq) {
//             for(int i=0; i<m; i++) {
//                 for(int j=0; j<n; j++) {
//                     dest[(i*np+p) + (j*nq+q)*lddest] = src[i+j*ldsrc];
//                 }
//             }
//         }

//         // local part of global submatrix A[i1:i2, j1:j2]
//         LMatrix<T> local_view(int i1, int j1, int i2, int j2) {
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
// //            printf("R%3d: i12={%d,%d}, j12={%d,%d}\n", grid().rank(), i12[0], i12[1], j12[0], j12[1]);
//             int mm = i12[1]-i12[0], nn = j12[1]-j12[0];

//             return LMatrix<T>(mm, nn, lld(), &data()[i12[0]+j12[0]*lld()]);
//         }

//         // replicate submatrix at A[i1:i2, j1:j2], left inclusive, right exclusive
//         LMatrix<T> replicate_in_all(int i1, int j1, int i2, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//             int mm = i12[1] - i12[0];
//             int nn = j12[1] - j12[0];

//             LMatrix<T> result(i2-i1, nn);
//             {             // stage 1: replicate in columns;
//                 LMatrix<T> sendbuf(mm, nn);
//                 copy_block(mm, nn, &data()[i12[0] + j12[0] * lld()], lld(), sendbuf.data(), sendbuf.ld());
//                 auto block_sizes = list_local_part_sizes(i1, i2, grid().dims()[0]);
//                 LMatrix<T> recvbuf(i2 - i1, nn);
//                 std::vector<int> displacements(grid().dims()[0], 0);
//                 std::vector<int> recv_counts(grid().dims()[0], 0);

//                 for (int i = 1; i < displacements.size(); i++) {
//                     displacements[i] = displacements[i - 1] + block_sizes[i - 1] * nn;
//                     recv_counts[i] = block_sizes[i] * nn;
//                 }
//                 recv_counts[0] = block_sizes[0] * nn;
// #ifdef DEBUG_REPLICATE
//                 if (grid().ranks()[0] == 0) {
//                     fmt::print("Stage 1: P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1], recv_counts,
//                                displacements);
//                 } else {
// //                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
//                 }
// #endif

//                 {
//                     Tracer tracer("Allgatherv in stage 1");
//                     int err = MPI_Allgatherv(sendbuf.data(), mm * nn, convert_to_mpi_datatype<T>(),
//                                              recvbuf.data(), recv_counts.data(), displacements.data(),
//                                              convert_to_mpi_datatype<T>(),
//                                              grid().comms()[0]);
//                     assert(err == MPI_SUCCESS);
//                 }

//                 {
//                     auto C = result.data(), ldc = result.ld(), A = recvbuf.data();
//                     Tracer tracer("scatter in stage 1 in replicate_in_all");
//                     for (int pi = 0; pi < grid().dims()[0]; pi++) {
//                         auto i12 = global_to_local_index({i1, i2}, pi, grid().dims()[0], 0);
//                         int mm = i12[1] - i12[0];
//                         int dpi = displacements[pi];
//                         int np = grid().dims()[0];
//                         for (int j = 0; j < nn; j++) {
//                             int jmm = j*mm;
//                             for (int i = 0; i < mm; i++) {
//                                 int gi = (i + i12[0]) * np + pi;
// //                                result(gi - i1, j) = recvbuf.data()[dpi + i + jmm];
//                                 C[gi-i1+j*ldc] = A[dpi+i+jmm];
//                             }
//                         }
//                     }
//                 }
//             }

//             LMatrix<T> allresult(i2-i1, j2-j1);
//             {// stage 2: replicate in rows:
//                 mm = i2 - i1;
//                 auto block_sizes = list_local_part_sizes(j1, j2, grid().dims()[1]);
//                 LMatrix<T> recvbuf(mm, j2-j1);
//                 std::vector<int> displacements( grid().dims()[1], 0 );
//                 std::vector<int> recv_counts( grid().dims()[1], 0);

//                 for(int i=1; i<displacements.size(); i++) {
//                     displacements[i] = displacements[i-1] + block_sizes[i-1]*mm;
//                     recv_counts[i] = block_sizes[i]*mm;
//                 }
//                 recv_counts[0] = block_sizes[0]*mm;
// #ifdef DEBUG_REPLICATE
//                 if(grid().ranks()[1]==0) {
//                     fmt::print("Stage 2: P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1],recv_counts, displacements );
//                 } else {
// //                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
//                 }
// #endif
//                 {
//                     Tracer tracer("Allgatherv in stage 2");
//                     int err = MPI_Allgatherv(result.data(), mm * nn, convert_to_mpi_datatype<T>(),
//                                              recvbuf.data(), recv_counts.data(), displacements.data(),
//                                              convert_to_mpi_datatype<T>(),
//                                              grid().comms()[1]);
//                     assert(err == MPI_SUCCESS);
//                 }

//                 {
//                     Tracer tracer("scatter in stage 2 in replicate_in_all");
//                     auto C = allresult.data(); int ldc = allresult.ld();
//                     auto A = recvbuf.data(); int lda = recvbuf.ld();
//                     for (int pi = 0; pi < grid().dims()[1]; pi++) {
//                         int dpi = displacements[pi];
//                         auto j12 = global_to_local_index({j1, j2}, pi, grid().dims()[1], 0);
//                         int nn = j12[1] - j12[0];
//                         for (int j = 0; j < nn; j++) {
//                             int jmm = j*mm;
//                             int gj = (j + j12[0]) * grid().dims()[1] + pi;
//                             for (int i = 0; i < mm; i++) {
// //                                allresult(i, gj - j1) = recvbuf.data()[dpi + i + jmm];
//                                 C[i+(gj-j1)*ldc] = A[dpi+i+jmm];
//                             }
//                         }
//                     }
//                 }
//             }
//             return allresult;
//         }
//         LMatrix<T> replicate_in_all_1stage(int i1, int j1, int i2, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//             int mm = i12[1] - i12[0];
//             int nn = j12[1] - j12[0];
//             int np = grid().dims()[0], nq = grid().dims()[1];
//             int max_mm = (i2-i1 + np-1) / np;
//             int max_nn = (j2-j1 + nq-1) / nq;
//             assert(mm<=max_mm && nn <=max_nn);

//             LMatrix<T> recvbuf(max_mm, max_nn*grid().np());
//             LMatrix<T> result(i2-i1, j2-j1);
//             LMatrix<T> sendbuf(max_mm, max_nn);
//             copy_block(mm, nn, &data()[i12[0] + j12[0] * lld()], lld(), sendbuf.data(), sendbuf.ld());
//             auto block_sizes = list_local_part_sizes(i1, i2, grid().dims()[0]);
//             {
//                 Tracer tracer("Allgather in stage 1");
//                 auto mpi_type = convert_to_mpi_datatype<T>();
//                 int err = MPI_Allgather(sendbuf.data(), max_mm*max_nn, mpi_type, recvbuf.data(), max_mm*max_nn, mpi_type, grid().comm());
//                 assert(err == MPI_SUCCESS);
//             }
//             {

//                 Tracer tracer("scatter in stage 1 in replicate_in_all");
//                 for (int pi = 0; pi < grid().np(); pi++) {
//                     auto C = result.data(), ldc = result.ld(), A = &recvbuf.data()[max_mm*max_nn*pi];
//                     auto pq = grid().convert_to_pq(pi);
//                     auto i12 = global_to_local_index({i1, i2}, pq[0], grid().dims()[0], 0);
//                     auto j12 = global_to_local_index({j1, j2}, pq[1], grid().dims()[1], 0 );
//                     int mm = i12[1] - i12[0];
//                     int nn = j12[1] - j12[0];
//                     int np = grid().dims()[0];
//                     int nq = grid().dims()[1];
//                     for (int j = 0; j < nn; j++) {
//                         int ldrecv = recvbuf.ld();
//                         for (int i = 0; i < mm; i++) {
//                             int gi = (i + i12[0]) * np + pq[0];
//                             int gj = (j + j12[0]) * nq + pq[1];
// //                                result(gi - i1, j) = recvbuf.data()[dpi + i + jmm];
//                             C[gi-i1+(gj-j1)*ldc] = A[i+j*ldrecv];
//                         }
//                     }
//                 }
//             }


//             return result;
//         }

//         // row dim (dim[0]) are no longer distributed; instead it's replicated on all row processs
//         LMatrix<T> replicate_in_columns(int i1, int j1, int i2, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//             int mm = i12[1] - i12[0];
//             int nn = j12[1] - j12[0];

//             LMatrix<T> sendbuf(mm, nn);
//             copy_block(mm, nn, &data()[i12[0]+j12[0]*lld()], lld(), sendbuf.data(), sendbuf.ld());
//             auto block_sizes = list_local_part_sizes(i1, i2, grid().dims()[0]);
//             LMatrix<T> recvbuf(i2-i1, nn);
//             std::vector<int> displacements( grid().dims()[0], 0 );
//             std::vector<int> recv_counts( grid().dims()[0], 0);

//             for(int i=1; i<displacements.size(); i++) {
//                 displacements[i] = displacements[i-1] + block_sizes[i-1]*nn;
//                 recv_counts[i] = block_sizes[i]*nn;
//             }
//             recv_counts[0] = block_sizes[0]*nn;
// #ifdef DEBUG_REPLICATE
//             if(grid().ranks()[0]==0) {
//                 fmt::print("P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1],recv_counts, displacements );
//             } else {
// //                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
//             }
// #endif
//             {
//                 Tracer tracer("Allgatherv in replicate_in_columns");
//                 int err = MPI_Allgatherv(sendbuf.data(), mm * nn, convert_to_mpi_datatype<T>(),
//                                          recvbuf.data(), recv_counts.data(), displacements.data(),
//                                          convert_to_mpi_datatype<T>(),
//                                          grid().comms()[0]);
//                 assert(err == MPI_SUCCESS);
//             }
//             LMatrix<T> result(i2-i1, nn);
//             {
//                 Tracer tracer("scatter in replicate_in_columns");
//                 auto C = result.data();
//                 auto A = recvbuf.data();
//                 int np = grid().dims()[0];
//                 int ldresult = result.ld();
//                 for (int pi = 0; pi < grid().dims()[0]; pi++) {
//                     auto i12 = global_to_local_index({i1, i2}, pi, np, 0);
//                     int mm = i12[1] - i12[0];
//                     int dpi = displacements[pi];

//                     for (int j = 0; j < nn; j++) {
//                         int jmm = j*mm;
//                         int nppi = i12[0]*np + pi;
//                         for (int i = 0; i < mm; i++) {
//                             int gi = (i + i12[0]) * np + pi;
//                             //result(gi - i1, j) = recvbuf.data()[dpi + i + jmm];
//                             C[i*np+nppi-i1 + j*ldresult] = A[dpi + i + jmm];
//                         }
//                     }
//                 }
//             }
//             return result;
//         }

//         LMatrix<T> transpose_and_replicate_in_rows(int i1, int j1, int i2, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             int p = grid().ranks()[0], q = grid().ranks()[1];
//             int np = grid().dims()[0], nq = grid().dims()[1];
//             // printf("np::%d, nq::%d\n", np, nq);
//             assert(np == nq);
//             auto i12 = global_to_local_index({i1, i2}, p, np, 0);
//             auto j12 = global_to_local_index({j1, j2}, q, nq, 0);
//             int mm = i12[1] - i12[0], nn = j12[1] - j12[0];

//             auto ri12 = global_to_local_index({j1, j2}, p, np, 0);
//             auto rj12 = global_to_local_index({i1, i2}, q, nq, 0);
//             int rmm = ri12[1] - ri12[0], rnn = rj12[1] - rj12[0];

//             std::vector<T> buf(rmm*rnn);
//             int ldbuf = rmm;

//             if(p == q) {
//                 assert(ldbuf == nn);
//                 copy_transposed_block(mm, nn, &m_data[i12[0]+j12[0]*lld()], lld(), buf.data(), ldbuf);
//             } else {
//                 std::vector<T> sendbuf(mm*nn);
//                 copy_transposed_block(mm, nn, &m_data[i12[0]+j12[0]*lld()], lld(), sendbuf.data(), nn);
//                 MPI_Status status;
//                 int err =
//                 MPI_Sendrecv(sendbuf.data(), mm*nn, convert_to_mpi_datatype<T>(), m_grid.convert_to_linear(q,p), p+q,
//                              buf.data(), rmm*rnn, convert_to_mpi_datatype<T>(), m_grid.convert_to_linear(q,p), p+q,
//                              m_grid.comm(), &status);
//                 assert(err == MPI_SUCCESS);
//             }
//             DMatrix<T> At(grid(), dims()[1], dims()[0],buf.data(), ldbuf, j1, i1);
//             return At.replicate_in_rows(j1, i1, j2, i2);
//         }

//         // replicate in rows means the processes in the same row
//         // will return the same local matrix.
//         LMatrix<T> replicate_in_rows(int i1, int j1, int i2, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//             int mm = i12[1] - i12[0];
//             int nn = j12[1] - j12[0];

//             LMatrix<T> sendbuf(mm, nn);
//             copy_block(mm, nn, &data()[i12[0]+j12[0]*lld()], lld(), sendbuf.data(), sendbuf.ld());
//             auto block_sizes = list_local_part_sizes(j1, j2, grid().dims()[1]);
//             LMatrix<T> recvbuf(mm, j2-j1);
//             std::vector<int> displacements( grid().dims()[1], 0 );
//             std::vector<int> recv_counts( grid().dims()[1], 0);

//             for(int i=1; i<displacements.size(); i++) {
//                 displacements[i] = displacements[i-1] + block_sizes[i-1]*mm;
//                 recv_counts[i] = block_sizes[i]*mm;
//             }
//             recv_counts[0] = block_sizes[0]*mm;
// #ifdef DEBUG_REPLICATE
//             if(grid().ranks()[1]==0) {
//                 fmt::print("P[{},{}] recving {} at {}\n", grid().ranks()[0], grid().ranks()[1],recv_counts, displacements );
//             } else {
// //                fmt::print("P[{},{}] sending size {}x{} data\n", grid().ranks()[0], grid().ranks()[1],mm, nn);
//             }
// #endif
//             {
//                 Tracer tracer("Allgatherv in replicate_in_rows");
//                 int err = MPI_Allgatherv(sendbuf.data(), mm * nn, convert_to_mpi_datatype<T>(),
//                                          recvbuf.data(), recv_counts.data(), displacements.data(),
//                                          convert_to_mpi_datatype<T>(),
//                                          grid().comms()[1]);
//                 assert(err == MPI_SUCCESS);
//             }
//             LMatrix<T> result(mm,j2-j1);

//             {
//                 Tracer tracer("scatter in replicate_in_rows");
//                 for (int pi = 0; pi < grid().dims()[1]; pi++) {
//                     auto j12 = global_to_local_index({j1, j2}, pi, grid().dims()[1], 0);
//                     int nn = j12[1] - j12[0];
//                     for (int j = 0; j < nn; j++) {
//                         int gj = (j + j12[0]) * grid().dims()[1] + pi;
//                         for (int i = 0; i < mm; i++) {
// //                        int gi = (i+i12[0])*grid().dims()[0] + pi;
//                             result(i, gj - j1) = recvbuf.data()[displacements[pi] + i + j * mm];
//                         }
//                     }
//                 }
//             }
//             return result;
//         }

//         void dereplicate_in_columns(LMatrix<T> A, int i1, int j1, int i2, int j2) {
// //            Tracer tracer(__FUNCTION__ );
//             int mm = A.dims()[0], nn = A.dims()[1];
//             assert(mm == i2-i1);
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//             int np = grid().dims()[0], nq = grid().dims()[1];
//             int p = grid().ranks()[0];
//             for(int j=0; j<nn; j++) {
//                 for(int i=0; i<i12[1]-i12[0]; i++) {
// //                    (*this)( i12[0]+i,  (j12[0]+j) ) = A( (i+i12[0])*np+p - i1, j);
//                     data()[i12[0]+i + (j12[0]+j)*lld()] = A( (i+i12[0])*np+p - i1, j);
//                 }
//             }
//         }

//         void dereplicate_in_rows(LMatrix<T> A, int i1, int j1, int i2, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             int mm = A.dims()[0], nn = A.dims()[1];
//             assert(nn == j2-j1);
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//             int nq = grid().dims()[1];
//             int q = grid().ranks()[1];
//             for(int j=0; j<j12[1]-j12[0]; j++) {
//                 for(int i=0; i<mm; i++) {
//                     data()[i12[0]+i + (j12[0]+j)*lld()] = A( i, (j+j12[0])*nq+q-j1);
//                 }
//             }
//         }

//         void dereplicate_in_all(LMatrix<T> A, int i1, int j1, int i2, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             int mm = A.dims()[0], nn = A.dims()[1];
//             assert(mm == i2-i1 && nn == j2-j1);
//             auto i12 = global_to_local_index({i1,i2}, grid().ranks()[0], grid().dims()[0], 0);
//             auto j12 = global_to_local_index({j1,j2}, grid().ranks()[1], grid().dims()[1], 0);
//             int np = grid().dims()[0], nq = grid().dims()[1];
//             int p = grid().ranks()[0], q = grid().ranks()[1];
//             for(int j=0; j<j12[1]-j12[0]; j++) {
//                 for(int i=0; i<i12[1]-i12[0]; i++) {
//                     data()[i12[0]+i + (j12[0]+j)*lld()] = A( (i+i12[0])*np+p - i1, (j+j12[0])*nq+q-j1);
//                 }
//             }
//         }

//         // align: the process that element 0 is at;
//         static inline std::array<int,2> global_to_local_index(std::array<int,2> ij, int p, int np, int align) {
// //            printf("ij={%d,%d}, p=%d, np=%d, align=%d\n", ij[0], ij[1], p, np, align);
//             int gi = ij[0], gj = ij[1]; 
//             int li = ceil(1.0*(gi-(p-align+np)%np)/np);
//             int lj = floor(1.0*(gj-(p-align+np)%np+np-1)/np);
//             return {li, lj};
//         }

//         inline std::vector<int> list_local_part_sizes(int i1, int i2,  int np) {
//             std::vector<int> l(np, 0);
//             for(int i=0; i<np; i++) {
//                 auto ii = global_to_local_index({i1,i2}, i, np, 0);
//                 l[i] = ii[1]-ii[0];
//             }
//             return l;
//         }


//         LMatrix<T> copy_upper_to_lower(LMatrix<T> A, int j1, int j2) {
//             Tracer tracer(__FUNCTION__ );
//             assert(grid().dims()[0] == grid().dims()[1]);
//             int p = grid().ranks()[0], q = grid().ranks()[1];
//             auto lij = global_to_local_index({j1,j2}, p, grid().dims()[0], 0);
//             int mm = A.dims()[0];
//             int recv_nn = lij[1] - lij[0];
//             LMatrix<T> recvbuf(mm, recv_nn);
//             MPI_Status status;
//             int err = MPI_Sendrecv(A.data(),mm*A.dims()[1], convert_to_mpi_datatype<T>(), grid().convert_to_linear(q,p), q+p,
//                          recvbuf.data(), mm*recv_nn, convert_to_mpi_datatype<T>(), grid().convert_to_linear(q,p), q+p,
//                          grid().comm(), &status);
//             assert(err == MPI_SUCCESS);

//             return recvbuf;
//         }


//         void print_in_process_sequence(LMatrix<T> A, const std::string &title) {
//             for(int i=0; i<dims()[0]; i++) {
//                 for(int j=0; j<dims()[1]; j++) {
//                     if(grid().ranks()[0] == i && grid().ranks()[1] == j) {
//                         A.print(fmt::format("P[{},{}]: {}", grid().ranks()[0], grid().ranks()[1], title));
//                     }
//                     grid().barrier();
//                 }
//             }
//         }


//         void permute_rows_inverse_ipiv(int i1, int j1, int i2, int j2, int* ipiv, int n_ipiv) {
//             Tracer tracer(__FUNCTION__);
//             for (int i = i1; i < i2; i++)
//                 m_permutation[i] = i;
//             for (int i = i1; i < i1 + n_ipiv; i++) {
//                 std::swap(m_permutation[i],
//                           m_permutation[ipiv[i - i1] - 1 + i1]);  // ipiv uses Fortran 1-base array index.
//             }
//             auto inv_perm = m_permutation;
//             for(int i=0; i<n_ipiv; i++) {
//                 inv_perm[m_permutation[i]] = i;
//             }
//             if(grid().rank()==0) {
// //                fmt::print("perm={}\n", m_permutation);
// //                fmt::print("inverse perm={}\n", inv_perm);
//             }
//             grid().barrier();
//             permute_rows_perm(i1, j1, i2, j2, inv_perm.data(), n_ipiv);

//         }

//         void permute_rows_ipiv(int i1, int j1, int i2, int j2, int* ipiv, int n_ipiv) {
//             Tracer tracer(__FUNCTION__);
//             for (int i = i1; i < i2; i++)
//                 m_permutation[i] = i;
//             for (int i = i1; i < i1 + n_ipiv; i++) {
// //                fmt::print("i={}, ipiv[i-i1]-1+i1={}\n", i, ipiv[i-i1]-1+i1);
//                 std::swap(m_permutation[i],
//                           m_permutation[ipiv[i - i1] - 1 + i1]);  // ipiv uses Fortran 1-base array index.
//             }
//             permute_rows_perm(i1, j1, i2, j2, m_permutation.data(), n_ipiv);

//         }
// //            fmt::print("ipiv={}, i1,i2={},{}\nm_permutations={}\n", ipiv, i1, i2, std::vector<int>(m_permutation.begin()+i1, m_permutation.begin()+i2));
//         void permute_rows_perm(int i1, int j1, int i2, int j2, int* perm, int n_perm){
//             auto j12 = global_to_local_index({j1, j2}, grid().ranks()[1], grid().dims()[1], 0);
//             auto i12 = global_to_local_index({i1, i2}, grid().ranks()[0], grid().dims()[0], 0);
//             int nn = j12[1] - j12[0], mm = i12[1] - i12[0];
//             int p = grid().ranks()[0], np = grid().dims()[0];
//             std::vector<int> recv_process_list, send_process_list;

//             // local rows.
//             struct PermInfo {
//                 int process;
//                 int local_index;
//             };
// //            if(grid().ranks()[1]==0){
// //            if (grid().ranks()[0] == 0) fmt::print("nn={}, ipiv={}, m_perm={}\n", nn, ipiv, m_permutation);
//             std::vector<int> send_counts(np, 0), recv_counts(np, 0);
// //            fmt::print("Process {}\n", grid().ranks()[0]);
//             std::vector<PermInfo> recv_info;
//             for (int i = i12[0]; i < i12[1]; i++) {
//                 int gi = i * np + p;
//                 int recv = perm[gi];
//                 if (recv != gi) {
// //                    fmt::print("li={} gi={} needs to recv {} from process {}\n", i, gi, recv, recv % np);
//                     recv_counts[recv % np]++;
//                     recv_info.push_back({recv % np, i});
//                 }
//             }
// //            fmt::print("recv_counts={}\n", recv_counts);
//             std::vector<PermInfo> send_info;
//             for (int i = i1; i < i2; i++) {
//                 if (perm[i] % np == p) {
//                     if (i != perm[i]) {
// //                        fmt::print("need to send li={},gi={} to process {} at gi={}\n",(m_permutation[i] - p) / np, m_permutation[i], i % np, i);
//                         send_counts[i % np]++;
//                         send_info.push_back({i % np, (perm[i] - p) / np});
//                     }
//                 }
//             }
// //            fmt::print("send_coutns={}\n", send_counts);
//             auto comp_by_first = [](PermInfo a, PermInfo b) { return a.process < b.process; };
//             std::stable_sort(send_info.begin(), send_info.end(), comp_by_first);
//             std::stable_sort(recv_info.begin(), recv_info.end(), comp_by_first);
// //                    fmt::print("sorted\nsend_info={}\nrecv_info={}\n", send_info, recv_info);

//             int send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0);
//             std::vector<int> send_counts_scan(np);
//             std::exclusive_scan(send_counts.begin(), send_counts.end(), send_counts_scan.begin(), 0);
// //            fmt::print("send_counts_scan={}\n", send_counts_scan);
//             LMatrix<T> sendbuf(send_count * nn, 1);
//             T *sbuf = sendbuf.data();
//             {
//                 Tracer tracer("copy to sendbuf in permute rows");
//                 int ldm = lld();
//                 for (int ii = 0; ii < send_info.size(); ii++){
//                     int offset = ldm * j12[0] + send_info[ii].local_index;
// #pragma GCC ivdep
//                     for (int i = 0; i < nn; i++) {
// //                        auto li = send_info[ii].local_index;
//                         sbuf[i + nn * ii] = m_data[offset + i * ldm];
//                     }

// //                  for (int i = 0; i < nn; i++) sbuf[i + nn * ii] = m_data[i];
//                 }
//             }

//             for (int i = 0; i < send_counts.size(); i++) {
//                 send_counts[i] *= nn;
//                 send_counts_scan[i] *= nn;
//             }

//             int recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
//             std::vector<int> recv_counts_scan(np);
//             std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_counts_scan.begin(), 0);
// //            fmt::print("recv_counts_scan={}\n", recv_counts_scan);
//             LMatrix<T> recvbuf(recv_count * nn, 1);
// //                    T* rbuf = recvbuf.data();

//             for (int i = 0; i < recv_counts.size(); i++) {
//                 recv_counts[i] *= nn;
//                 recv_counts_scan[i] *= nn;
//             }
//             {
//                 Tracer tracer("Alltoallv in permute rows");
//                 MPI_Alltoallv(sendbuf.data(), send_counts.data(), send_counts_scan.data(), convert_to_mpi_datatype<T>(),
//                               recvbuf.data(), recv_counts.data(), recv_counts_scan.data(), convert_to_mpi_datatype<T>(),
//                               grid().comms()[0]);
//             }

//             T *rbuf = recvbuf.data();
//             {
//                 Tracer tracer("copy from recvbuf in permute rows");
//                 for (int ii = 0; ii < recv_info.size(); ii++) {
//                     auto li = recv_info[ii].local_index;
// #pragma GCC ivdep
//                     for (int i = 0; i < nn; i++)
//                         m_data[li + (i + j12[0]) * lld()] = rbuf[i + nn * ii];
//                 }
//             }


//         }

//         void swap_rows(int row1, int row2, int j1, int j2) {
//             int p = grid().ranks()[0], q = grid().ranks()[1], np = grid().dims()[0], nq = grid().dims()[1];
//             auto j12 = global_to_local_index({j1, j2}, q, nq, 0); // must swap the whole row;

//             if(row1 % np == row2 % np && row1 % np == p) { // local swap rows
//                 int l1 = (row1 - p) / np, l2 = (row2 - p) / np;  // local row inices
//                 for(int jj = j12[0]; jj < j12[1]; jj++) {
//                     std::swap(data()[l1 + jj * lld()], data()[l2 + jj * lld()]);
//                 }
//             } else { // cross process swap rows
//                 int nn = j12[1] - j12[0];
//                 std::vector<T> recvbuf(nn);
//                 std::vector<T> sendbuf(nn);
//                 MPI_Status status;
// //                if (q==0) fmt::print("P{} and P{} are swapping row {} and {}\n", row1 % np, row2 % np, row1, row2);
//                 if (p == row1 % np) {
//                     int l1 = (row1 - p) / np;
//                     for(int jj=0; jj<nn; jj++) {
//                         sendbuf[jj] = data()[l1 + (j12[0] + jj) * lld()];
//                     }
// //                        fmt::print("P{} sending to P{} and recv'ing from P{}\n", p, row2 % np, row2 % np);
//                     MPI_Sendrecv(sendbuf.data(), nn,
//                                  convert_to_mpi_datatype<T>(), row2 % np, row1,
//                                  recvbuf.data(), nn, convert_to_mpi_datatype<T>(), row2 % np,
//                                  row2, grid().comms()[0], &status);
//                     for(int jj=0; jj<nn; jj++) {
//                         data()[l1 + (j12[0] + jj) * lld()] = recvbuf[jj];
//                     }
//                 } else if (p == row2 % np) {
//                     int l2 = (row2 - p) / np;
//                     for(int jj=0; jj<nn; jj++) {
//                         sendbuf[jj] = data()[l2 + (j12[0] + jj) * lld()];
//                     }
// //                        fmt::print("P{} sending to P{} and recv'ing from P{}\n", p, row1 % np, row1 % np);
//                     MPI_Sendrecv(sendbuf.data(), nn,
//                                  convert_to_mpi_datatype<T>(), row1 % np, row2,
//                                  recvbuf.data(), nn, convert_to_mpi_datatype<T>(), row1 % np, row1,
//                                  grid().comms()[0], &status);
//                     for(int jj=0; jj<recvbuf.size(); jj++) {
//                         data()[l2+ (j12[0] + jj) * lld()] = recvbuf[jj];
//                     }
//                 }


//             }
//         }

//         void substract(DMatrix<T>& A) {
//             auto i12 = global_to_local_index({0, A.dims()[0]}, A.grid().ranks()[0], A.grid().dims()[0], 0);
//             auto j12 = global_to_local_index({0, A.dims()[1]}, A.grid().ranks()[1], A.grid().dims()[1], 0);
//             for(int jj=j12[0]; jj<j12[1]; jj++) {
//                 for(int ii=i12[0]; ii<i12[1]; ii++) {
//                     m_data[ii+jj*lld()] -= A.data()[ii+jj*A.lld()];
//                 }
//             }
//         }




//     };


//     template<typename T> size_t LMatrix<T>::total_allocated_bytes {0};
//     template<typename T> size_t LMatrix<T>::maximum_allocated_bytes {0};

//     template<typename T>
//     struct DMatrixDescriptor {
//         DMatrix<T> matrix;
//         int i1, j1, i2, j2;
//         int m() {
//             return i2 - i1;
//         }
//         int n() {
//             return j2 - j1;
//         }
//         LMatrix<T> replicate_in_rows() {
//             return matrix.replicate_in_rows(i1, j1, i2, j2);
//         }
//         LMatrix<T> replicate_in_columns() {
//             return matrix.replicate_in_columns(i1, j1, i2, j2);
//         }
//         LMatrix<T> replicate_in_all() {
//             return matrix.replicate_in_all(i1, j1, i2, j2);
//         }
//         void print(const std::string &title) {
//             auto A = replicate_in_all();
//             if(matrix.grid().rank()==0) A.print(title);
//         }
//     };

// }


