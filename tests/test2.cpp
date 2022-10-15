#include "darray.h"
//#include "lapack_like.h"
#include "cblas_like.h"
#include <iostream>

using namespace std;
int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);
    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q);
//    if (p!=q) {
//        cout << " p != q! " << p << " , " << q << endl;
//        return 0;
//    }
    DArray::Grid g(MPI_COMM_WORLD, p, q);

    DArray::DMatrix<float> m(g, 10,10);

    m.set_function([](int gi, int gj) ->float {
        return gi+gj/10.0;
    });

    m.collect_and_print("matrix m", 0);
    // m.print_by_process("matrix m");

    m.grid().barrier();

    if (g.rank()==0){
        auto ij = DArray::DMatrix<float>::global_to_local_index({4,5}, 1, 3, 1); 
        cout << "ij = " << ij[0] << "," << ij[1] << endl;
    }

    int i1 = 0, j1 =0 , i2 = 6, j2 = 7;

   for(int i=0; i<m.grid().np(); i++) {
       if(m.grid().rank()==i){
           auto local = m.local_view(i1, j1, i2, j2);
           char buf[50];
           sprintf(buf, "R[%2d,%2d]: localview of %d:%d, %d:%d", m.grid().ranks()[0], m.grid().ranks()[1], i1, i2, j1,
                   j2);
           local.print(buf);
       }
       m.grid().barrier();
   }



    // auto rpl_col = m.replicate_in_columns(i1, j1, i2, j2);
    // if (g.ranks()[0]==1) {
    //     for (int j = 0; j < g.dims()[1]; j++) {
    //         if (g.ranks()[1]==j)
    //             rpl_col.print(fmt::format("rep_col of column {} of A[{}:{},{}:{}]", g.ranks()[1], i1, i2, j1, j2));
    //         g.barrier_row();
    //     }
    // }
    // g.barrier();


//    for(int i=0; i<rpl_col.dims()[0]; i++) {
//        for(int j=0; j<rpl_col.dims()[1]; j++) {
//            rpl_col(i,j) -= (i+i1) - 100;
//        }
//    }

//    m.dereplicate_in_columns(rpl_col, i1, j1, i2, j2);

//    m.collect_and_print("matrix m after dereplicate_in_columns", 0);

//     auto rpl_row = m.replicate_in_rows(i1, j1, i2, j2);
//    if (g.ranks()[1]==1) {
//        for (int j = 0; j < g.dims()[0]; j++) {
//            if (g.ranks()[0] == j)
//                rpl_row.print(fmt::format("rep_row of row {} of A[{}:{},{}:{}]", g.ranks()[0],
//                                          i1, i2, j1, j2));
//            g.barrier_col();
//        }
//    }

//    for(int i=0; i<rpl_row.dims()[0]; i++) {
//        for(int j=0; j<rpl_row.dims()[1]; j++) {
//            rpl_row(i,j) -= (j+j1)/10.0 - 0.787;
//        }
//    }
//    m.dereplicate_in_rows(rpl_row, i1, j1, i2, j2);
//    m.collect_and_print(fmt::format("matrix m after dereplicate_in_rows {}:{}x{}:{}",i1,i2,j1,j2), 0);

//     auto rpl_all = m.replicate_in_all(i1, j1, i2, j2);
// //    if (g.rank() == 0) {
// //        rpl_all.print(fmt::format("rep_all of  of A[{}:{},{}:{}]",
// //                                  i1, i2, j1, j2));
// //    }
//     for(int i=0; i<rpl_all.dims()[0]; i++) {
//         for(int j=0; j<rpl_all.dims()[1]; j++) {
//             rpl_all(i,j) -= 2*(j+j1)/10.0 + 2*(i+i1);
//         }
//     }
//     m.dereplicate_in_all(rpl_all, i1, j1, i2, j2);
//     m.collect_and_print(fmt::format("matrix m after dereplicate_in_all {}:{}x{}:{}",i1,i2,j1,j2), 0);

    MPI_Finalize();
}
