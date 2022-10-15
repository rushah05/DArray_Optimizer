#include "darray.h"
#include "read_input.h"
#include "lapack_like.h"
#include "cblas_like.h"
#include "dkernel.h"

extern void cuda_init();
extern void cuda_finalize();

template<typename T>
T gaussrand(){
    static T V1, V2, S;
    static int phase = 0;
    T X;
    
    if(phase == 0) {
        do {
            T U1 = (T)rand() / RAND_MAX;
            T U2 = (T)rand() / RAND_MAX;
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    
    phase = 1 - phase;
    return X;
}


int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int np, p, q;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  DArray::Grid::np_to_pq(np, p, q); // np = p*q
  DArray::Grid g(MPI_COMM_WORLD, p, q);
  g.barrier();
  cuda_init();
  DArray::ElapsedTimer timer;

  if(argc < 7){
      if(g.rank() == 0) printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(float)> <hyper parameter C> <power refinement q(int)>\n");
      return 0;
  }

  char *filename = argv[1];
  // long long int n = atoi(argv[2]);
  int n = atoi(argv[2]);
  int d = atoi(argv[3]);
  int k = atoi(argv[4]); /*no of columns of the tall matrix A, B and Omega*/
  float gamma = strtod(argv[5], NULL);
  int qq = atoi(argv[6]);
  float C = strtod(argv[7], NULL);

  if(qq < 1) qq=1;

  if(qq*k > n/2) {
    if(g.rank()==0) printf(" The kernel needs to be tall and skinny\n", p, q, n, d, k);
    return 0;
  }

  if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, n::%d, d::%d, k::%d) \n", p, q, n, d, k);
  float *x=(float*)malloc(sizeof(float)*d*n);
  float *y=(float*)malloc(sizeof(float)*n);
  for(int i=0; i<d; i++){
    for(int j=0; j<n; j++){
      x[i+j*d]=0.0;
    }
  }
  for(int i=0; i<n; ++i){
    y[i]=0.0;
  }
  

  timer.start();
  read_input_file<float>(g.rank(), filename, n, d, x, d, y);
  int ms = timer.elapsed();
  if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  g.barrier();

  DArray::DMatrix<float> X(g, d, n);
  DArray::DMatrix<float> Y(g, 1, n);
  X.set_value_from_matrix(x, d);
  Y.set_value_from_matrix(y, 1);
  free(x);
  free(y);
  g.barrier();


  if(g.rank()==0) fmt::print("Starting low rank approximation\n");
  DArray::DMatrix<float> A(g, n, k);
  {
    DArray::DMatrix<float> O(g, n, k);
    for(int i=0; i<n; i++){
      for(int j=0; j<k; j++){
        O.at(i,j)=gaussrand<float>();
      }
    }
    timer.start();
    DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, O, A, gamma);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: LRA (low-rank approximation of kernel) - K({},{}) ~ G({},{}): {}(ms), memory consumption: {:.1f}MBytes\n",  g.ranks()[0], g.ranks()[1], n, n, n, k, ms, DArray::getPeakRSS()/1.0e6);
  }
  g.barrier();

  cuda_finalize();
  MPI_Finalize();
}















// #include "darray.h"
// #include "read_input.h"
// #include "lapack_like.h"
// #include "cblas_like.h"
// #include "dkernel.h"
// #include "dbarrier.h"

// // extern void cuda_init();
// // extern void cuda_finalize();
// // // void extern TC_GEMM(char *a, char *b, int m, int n, int k, float alpha, float beta, float *fA, float *fB, float *fC);

// template<typename T>
// T gaussrand(){
//     static T V1, V2, S;
//     static int phase = 0;
//     T X;
    
//     if(phase == 0) {
//         do {
//             T U1 = (T)rand() / RAND_MAX;
//             T U2 = (T)rand() / RAND_MAX;
            
//             V1 = 2 * U1 - 1;
//             V2 = 2 * U2 - 1;
//             S = V1 * V1 + V2 * V2;
//         } while(S >= 1 || S == 0);
        
//         X = V1 * sqrt(-2 * log(S) / S);
//     } else
//         X = V2 * sqrt(-2 * log(S) / S);
    
//     phase = 1 - phase;
//     return X;
// }


// int main(int argc, char** argv){
//   MPI_Init(&argc, &argv);
//   int np, p, q;
//   MPI_Comm_size(MPI_COMM_WORLD, &np);
//   DArray::Grid::np_to_pq(np, p, q); // np = p*q
//   DArray::Grid g(MPI_COMM_WORLD, p, q);
//   g.barrier();
//   // cuda_init();
//   DArray::ElapsedTimer timer;

//   if(argc < 9){
//       if(g.rank() == 0) printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(float)> <power refinement q(int)>\n");
//       return 0;
//   }

//   char *filename = argv[1];
//   // long long int n = atoi(argv[2]);
//   int n = atoi(argv[2]);
//   int d = atoi(argv[3]);
//   int k = atoi(argv[4]); /*no of columns of the tall matrix A, B and Omega*/
//   int bs = atoi(argv[5]); 
//   float gamma = strtod(argv[6], NULL);
//   int qq = atoi(argv[7]);
//   float C = strtod(argv[8], NULL);

//   if(qq < 1) qq=1;

//   if(bs <= 0) bs = 128;

//   if(qq*k > n/2) {
//     if(g.rank()==0) printf(" The kernel needs to be tall and skinny\n", p, q, n, d, k);
//     return 0;
//   }

//   if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, n::%d, d::%d, k::%d) \n", p, q, n, d, k);
//   float *x=(float*)malloc(sizeof(float)*d*n);
//   float *y=(float*)malloc(sizeof(float)*n);
//   for(int i=0; i<d; i++){
//     for(int j=0; j<n; j++){
//       x[i+j*d]=0.0;
//     }
//   }
//   for(int i=0; i<n; ++i){
//     y[i]=0.0;
//   }
  
//   if(g.rank() == 0) {
//     timer.start();
//     read_input_file<float>(g.rank(), filename, n, d, x, d, y);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   MPI_Bcast(x, d*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
//   MPI_Bcast(y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
//   g.barrier();

//   DArray::DMatrix<float> X(g, d, n);
//   DArray::DMatrix<float> Y(g, 1, n);
//   X.set_value_from_matrix(x, d);
//   Y.set_value_from_matrix(y, 1);
//   free(x);
//   free(y);
//   g.barrier();


//   DArray::DMatrix<float> A(g, n, k);
//   A.set_constant(0.0);
//   {
//     timer.start();
//     DArray::DMatrix<float> O(g, n, k);
//     for(int i=0; i<n; i++){
//       for(int j=0; j<k; j++){
//         O.at(i,j)=gaussrand<float>();
//       }
//     } 

//     DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, O, A, gamma, bs);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: LRA (low-rank approximation of kernel) - K({},{}) ~ G({},{}): {}(ms)\n",  g.ranks()[0], g.ranks()[1], n, n, n, k, ms);
//   }
//   // A.collect_and_print("A");
//   g.barrier();


//   DArray::DMatrix<float> K(g, n, qq*k);
//   K.set_constant(0.0);
//   {
//     timer.start();
//     DArray::DMatrixDescriptor<float> Kdesc{K, 0, 0, n, qq*k}, Adesc{A, 0, 0, n, k};
//     auto Kl=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j2);
//     auto Al=Adesc.matrix.local_view(Adesc.i1, Adesc.j1, Adesc.i2, Adesc.j2);
//     DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), Kl.data(), Kl.ld());
    
//     DArray::DMatrix<float> Atemp(g, n, k);
//     // if(g.rank() == 0) printf("Kl[%d,%d] - (%d,%d,%d,%d) \n", Kl.dims()[0], Kl.dims()[1], Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j2);
//     for(int i=2; i<=qq; ++i){
//       // Atemp.set_value_from_matrix(A.data(), A.lld());
//       Atemp=A.clone();
//       A.set_constant(0);
//       DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, Atemp, A, gamma, bs);
//       // if(g.rank() == 0) printf("Al[%d,%d], Kl[%d,%d] - (%d,%d,%d,%d) \n", Al.dims()[0], Al.dims()[1], Kl.dims()[0], Kl.dims()[1], Kdesc.i1, (Kdesc.j1+(i-1)*k)-1, Kdesc.i2, Kdesc.j1+(i*k));
//       DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*k], Kl.ld());
//     }
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: Creating a Krylov subspace... ({}, {}): {}(ms)\n",  g.ranks()[0], g.ranks()[1], n, qq*k, ms);
//   }
//   // K.collect_and_print("K");
//   g.barrier();


//   // Distributed QR
//   DArray::DMatrix<float> Q = K.clone();
//   {
//     timer.start();
//     tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();
//   // Q.collect_and_print("Q");

//   // Distributed KQ
//   DArray::DMatrix<float> KQ(g, n, qq*k);
//   KQ.set_constant(0.0);
//   {
//     timer.start();
//     DArray::LRA<float>(g.rank(), np, g, n, d, qq*k, X, Y, Q, KQ, gamma, bs);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();

//   // KQ.collect_and_print("KQ");

//   DArray::DMatrix<float> CC(g, qq*k, qq*k);
//   CC.set_constant(0.0);
//   {
//     timer.start();
//     DArray::DMatrixDescriptor<float> Ql {Q, 0, 0, n, qq*k};
//     DArray::DMatrixDescriptor<float> KQl {KQ, 0, 0, n, qq*k};
//     DArray::DMatrixDescriptor<float> CCl {CC, 0, 0, qq*k, qq*k};
//     DArray::matrix_multiply("T", "N", Ql, KQl, CCl, 1.0f, 0.0f);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: C=QT*KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();
//   // CC.collect_and_print("CC");

//   DArray::DMatrix<float> U(g, qq*k, qq*k);
//   U.set_constant(0.0);
//   {
//     timer.start();
//     auto Ul=DArray::svd<float>(qq*k, CC, 0, 0, qq*k, qq*k);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: SVD takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//     U.dereplicate_in_all(Ul, 0, 0, qq*k, qq*k);
//   }
//   g.barrier();
//   // U.collect_and_print("U");

//   DArray::DMatrix<float> Z(g, n, k);
//   Z.set_constant(0.0);
//   {
//     timer.start();
//     auto Ql=Q.replicate_in_rows(0, 0, n, qq*k);
//     auto Ul=U.replicate_in_columns(0, 0, qq*k, k);
//     auto Zl=Z.local_view(0, 0, n, k);
//     // if(g.rank()==0) printf("Ql[%d,%d], Ul[%d,%d], Zl[%d,%d]\n", Ql.dims()[0], Ql.dims()[1], Ul.dims()[0], Ul.dims()[1], Zl.dims()[0], Zl.dims()[1]);
//     float one=1.0f, zero=0.0f;
//     sgemm_("N", "N", &Ql.dims()[0], &Ul.dims()[1], &Ql.dims()[1], &one, Ql.data(), &Ql.ld(), Ul.data(), &Ul.ld(), &zero, Zl.data(), &Zl.ld());
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: Creating Low Rank Approximator Z takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();
//   // Z.collect_and_print("Z");

//   // DArray::Knorm<float>(g.rank(), np, g, n, d, k, X, Y, Z, gamma, bs);
  
//   // DArray::DMatrix<double> Zd(g, n, k);{
//   //   for(int i=0; i<n; ++i){
//   //     for(int j=0; j<k; ++j){
//   //       Zd.at(i,j)=Z.at(i,j);
//   //     }
//   //   }
//   // }

//   // cuda_finalize();  
//   // MPI_Finalize();
// }


