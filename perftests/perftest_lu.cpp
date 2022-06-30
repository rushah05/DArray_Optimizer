#include "darray.h"
#include "lapack_like.h"
#include "utils.h"
#include <iostream>
#include <map>
#include <time.h>
#include <stdlib.h>

using namespace std;
int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    int np, p, q;
    int  n = 10;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q);
    DArray::Grid g(MPI_COMM_WORLD, p, q);

    bool save_matrix_C = false;
    bool print_accum_tracer = false;
    bool save_events_to_file = false;
    bool no_check_correctness = false;
    DArray::ArgsParser parser;
//    parser.add_option(m, "matrix size m (rows)", "num-rows", 'm', "m");
    parser.add_option(n, "matrix size n (columns)", "num-cols", 'n', "n");
    parser.add_option(p, "process grid rows", "process-rows", 'p', "p");
    parser.add_option(q, "process grid columns", "process-columns", 'q', "q");
//    parser.add_option(save_matrix_C, "save matrix C in C.mat", "save-matrix-c", 'c');
    parser.add_option(print_accum_tracer, "print out the accumulated tracer times", "print-accum-tracer", 't');
    parser.add_option(save_events_to_file, "save events process to file named trace.log.<rank>", "save-events-to-file", 'e');
    parser.add_option(no_check_correctness, "disable residual checking for correctness", "no-check", 'h');
    parser.parse(argc, argv);

    if(g.rank() == 0)
        fmt::print("options: n={}, p={}, q={}, save_matrix_c={}, print_accum_tracer={}, save_events_to_file={}\n",
                   n, p, q, save_matrix_C, print_accum_tracer, save_events_to_file);

    if (g.rank()==0) printf("#### grid pxq %dx%d #### \n", p, q);

    g.barrier();
    fmt::print("after init MPI, memory consumption: {:.1f}MBytes\n", DArray::getPeakRSS()/1.0e6);

    DArray::DMatrix<float> A(g, n, n);
    A.set_uniform(-2,2);
    auto A0 = A.clone();
    auto A00 = A.clone();

//    {
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++)
//                A.at(i, j) =  std::min(i+1,j+1);
//        }
//    }
//    A.collect_and_print("A after write_at", 0);
    fmt::print("after init A, memory consumption: {:.1f}MBytes\n", DArray::getPeakRSS()/1.0e6);

    g.barrier();
    DArray::Tracer::trace_on();
//    DArray::Tracer::event_on();
    DArray::ElapsedTimer timer;
    timer.start();
    DArray::lu_factorize_partial_pivot(A, 0, 0, n, n);
    g.barrier();
    int ms = timer.elapsed();
    fmt::print("P[{},{}]: lu with partial pivoting takes: {}(ms) GFLOPS: {}\n",  g.ranks()[0], g.ranks()[1],ms, 2.0/3*n*n*n/ms/1.0e6);
    DArray::Tracer::trace_off();

    size_t peak_resident_memory = DArray::getPeakRSS();
    fmt::print("Peak memory consumption: {:.1f} MBytes\n", peak_resident_memory/1.0e6);
#if 0
    {
        int i = n-1, j = n-1;
        auto probe = A(i, j);
        fmt::print("A[{},{}]={}\n", i, j, probe);
//        if( DArray::relative_difference(probe,1) > 1e-5 ) {
//            fmt::print("\033[m31Incorrect Result1 A({},{})={}, should be 1\033[m0\n", i, j, probe);
//            return 1;
//        }
    }
#endif
    if(!no_check_correctness) {

        {
            DArray::Tracer tracer("tri_tri_mul", true);
            DArray::triangular_triangular_multiply(A, A, A0, 1.0f, 0.0f);
        }

        {
            DArray::Tracer tracer("final permute P*LU", true);
            A0.permute_rows_inverse_ipiv(0, 0, n, n, A.ipiv().data(), A.ipiv().size());
        }
        auto a00n = A00.fnorm();
        A00.substract(A0);
        auto diff_norm = A00.fnorm();
        if(g.rank()==0) fmt::print("norm(A-P*L*U)/norm(A)={}\n", diff_norm / a00n );
//        auto fn = A0.fnorm();
    }

    if (print_accum_tracer) {
        auto& events = DArray::Tracer::events;
        //fmt::print("events size: {}\n", events.size());
        std::map<const char*,long long> op_map;
        for(auto &event : events) {
            if(event.etype == DArray::Tracer::EventType::EndEvent) {
                op_map[event.op] += event.end - event.start;
            }
        }
        std::vector<std::pair<const char*,long long>> op_list(op_map.begin(), op_map.end());
        std::sort(op_list.begin(), op_list.end(), [](
                const pair<const char*,long long> &a,
                const pair<const char*,long long> &b) {
            return (a.second > b.second);
        });
        for (int i = 0; i < g.np(); i++) {
            if (g.rank() == i) {
                fmt::print("============ P[{}] ==========\n", g.rank());
                for (auto &pair : op_list) {
                    fmt::print("op: {}, accum time in ms: {}\n", pair.first, pair.second);
                }
            }
        }
    }

    if(save_events_to_file) {
        DArray::Tracer::save_events_to_file(fmt::format("trace.log.{}", g.rank()));
    }


    MPI_Finalize();
}