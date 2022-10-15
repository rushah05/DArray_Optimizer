







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
  
  // if(g.rank() == 0) {
    timer.start();
    read_input_file<float>(g.rank(), filename, n, d, x, d, y);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // }
  // MPI_Bcast(x, d*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Bcast(y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
  A.set_constant(0.0);
  {
    timer.start();
    DArray::DMatrix<float> O(g, n, k);
    for(int i=0; i<n; i++){
      for(int j=0; j<k; j++){
        O.at(i,j)=gaussrand<float>();
      }
    } 

    DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, O, A, gamma);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: LRA (low-rank approximation of kernel) - K({},{}) ~ G({},{}): {}(ms)\n",  g.ranks()[0], g.ranks()[1], n, n, n, k, ms);
  }
  // A.collect_and_print("A");
  g.barrier();


  DArray::DMatrix<float> K(g, n, qq*k);
  K.set_constant(0.0);
  {
    timer.start();
    DArray::DMatrixDescriptor<float> Kdesc{K, 0, 0, n, qq*k}, Adesc{A, 0, 0, n, k};
    auto Kl=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j2);
    auto Al=Adesc.matrix.local_view(Adesc.i1, Adesc.j1, Adesc.i2, Adesc.j2);
    DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), Kl.data(), Kl.ld());
    
    DArray::DMatrix<float> Atemp(g, n, k);
    // if(g.rank() == 0) printf("Kl[%d,%d] - (%d,%d,%d,%d) \n", Kl.dims()[0], Kl.dims()[1], Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j2);
    for(int i=2; i<=qq; ++i){
      // Atemp.set_value_from_matrix(A.data(), A.lld());
      Atemp=A.clone();
      A.set_constant(0);
      DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, Atemp, A, gamma);
      // if(g.rank() == 0) printf("Al[%d,%d], Kl[%d,%d] - (%d,%d,%d,%d) \n", Al.dims()[0], Al.dims()[1], Kl.dims()[0], Kl.dims()[1], Kdesc.i1, (Kdesc.j1+(i-1)*k)-1, Kdesc.i2, Kdesc.j1+(i*k));
      DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*k], Kl.ld());
    }
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: Creating a Krylov subspace when q={}... ({}, {}): {}(ms)\n",  qq, g.ranks()[0], g.ranks()[1], n, qq*k, ms);
  }
  // K.collect_and_print("K");
  g.barrier();


  // Distributed QR
  DArray::DMatrix<float> Q = K.clone();
  {
    timer.start();
    tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  g.barrier();
  // Q.collect_and_print("Q");

  // Distributed KQ
  DArray::DMatrix<float> KQ(g, n, qq*k);
  KQ.set_constant(0.0);
  {
    timer.start();
    DArray::LRA<float>(g.rank(), np, g, n, d, qq*k, X, Y, Q, KQ, gamma);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  g.barrier();

  // KQ.collect_and_print("KQ");

  DArray::DMatrix<float> CC(g, qq*k, qq*k);
  CC.set_constant(0.0);
  {
    timer.start();
    DArray::DMatrixDescriptor<float> Ql {Q, 0, 0, n, qq*k};
    DArray::DMatrixDescriptor<float> KQl {KQ, 0, 0, n, qq*k};
    DArray::DMatrixDescriptor<float> CCl {CC, 0, 0, qq*k, qq*k};
    DArray::matrix_multiply("T", "N", Ql, KQl, CCl, 1.0f, 0.0f);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: C=QT*KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  g.barrier();
  // CC.collect_and_print("CC");

  DArray::DMatrix<float> U(g, qq*k, qq*k);
  U.set_constant(0.0);
  {
    timer.start();
    auto Ul=DArray::svd<float>(qq*k, CC, 0, 0, qq*k, qq*k);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: SVD takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    U.dereplicate_in_all(Ul, 0, 0, qq*k, qq*k);
  }
  g.barrier();
  // U.collect_and_print("U");

  DArray::DMatrix<float> Z(g, n, k);
  Z.set_constant(0.0);
  {
    timer.start();
    auto Ql=Q.replicate_in_rows(0, 0, n, qq*k);
    auto Ul=U.replicate_in_columns(0, 0, qq*k, k);
    auto Zl=Z.local_view(0, 0, n, k);
    // if(g.rank()==0) printf("Ql[%d,%d], Ul[%d,%d], Zl[%d,%d]\n", Ql.dims()[0], Ql.dims()[1], Ul.dims()[0], Ul.dims()[1], Zl.dims()[0], Zl.dims()[1]);
    float one=1.0f, zero=0.0f;
    sgemm_("N", "N", &Ql.dims()[0], &Ul.dims()[1], &Ql.dims()[1], &one, Ql.data(), &Ql.ld(), Ul.data(), &Ul.ld(), &zero, Zl.data(), &Zl.ld());
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: Creating Low Rank Approximator Z takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  g.barrier();
  // Z.collect_and_print("Z");

  // timer.start();
  // DArray::Knorm<float>(g.rank(), np, g, n, d, k, X, Y, Z, gamma);
  // ms = timer.elapsed();
  // if(g.rank()==0) fmt::print("P[{},{}]: Calculating norm(K-ZZT) takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);

  cuda_finalize();
  MPI_Finalize();
}






















// #include "darray.h"
// #include "read_input.h"
// #include "lapack_like.h"
// #include "cblas_like.h"
// // #include "dkernel.h"
// // #include "dbarrier.h"

// extern void cuda_init();
// extern void cuda_finalize();
// extern void LRA(int rk, int n, int d, int k, float *X, int ldx, float *Y, float *O, int ldo, float *A, int lda, float gamma, int bs=8192);

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
//   cuda_init();
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

//   auto XX=X.replicate_in_all(0, 0, d, n).transpose();
//   auto YY=Y.replicate_in_all(0, 0, 1, n).transpose();
//   // if(g.rank() == 0) YY.print("YY");

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

//     auto OO=O.replicate_in_all(0, 0, n, k);
//     auto AA=A.replicate_in_all(0, 0, n, k);
//     LRA(g.rank(), n, d, k, XX.data(), XX.ld(), YY.data(), OO.data(), OO.ld(), AA.data(), AA.ld(), gamma, bs);
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
    
//     DArray::DMatrix<float> Atmp(g, n, k);
//     // if(g.rank() == 0) printf("Kl[%d,%d] - (%d,%d,%d,%d) \n", Kl.dims()[0], Kl.dims()[1], Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j2);
//     for(int i=2; i<=qq; ++i){
//       // Atemp.set_value_from_matrix(A.data(), A.lld());
//       Atmp=A.clone();
//       A.set_constant(0.0);
//       auto AAtmp=Atmp.replicate_in_all(0, 0, n, k);
//       auto AA=A.replicate_in_all(0, 0, n, k);
//       LRA(g.rank(), n, d, k, XX.data(), XX.ld(), YY.data(), AAtmp.data(), AAtmp.ld(), AA.data(), AA.ld(), gamma, bs);
//       // if(g.rank() == 0) printf("Al[%d,%d], Kl[%d,%d] - (%d,%d,%d,%d) \n", Al.dims()[0], Al.dims()[1], Kl.dims()[0], Kl.dims()[1], Kdesc.i1, (Kdesc.j1+(i-1)*k)-1, Kdesc.i2, Kdesc.j1+(i*k));
//       DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*k], Kl.ld());
//     }
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: Creating a Krylov subspace... ({}, {}): {}(ms)\n",  g.ranks()[0], g.ranks()[1], n, qq*k, ms);
//   }
//   // K.collect_and_print("K");
//   g.barrier();


//   // // Distributed QR
//   // DArray::DMatrix<float> Q = K.clone();
//   // {
//   //   timer.start();
//   //   tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   // }
//   // g.barrier();
//   // // Q.collect_and_print("Q");

//   // // Distributed KQ
//   // DArray::DMatrix<float> KQ(g, n, qq*k);
//   // KQ.set_constant(0.0);
//   // {
//   //   timer.start();
//   //   DArray::LRA<float>(g.rank(), np, g, n, d, qq*k, X, Y, Q, KQ, gamma, bs);
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   // }
//   // g.barrier();

//   // // KQ.collect_and_print("KQ");

//   // DArray::DMatrix<float> CC(g, qq*k, qq*k);
//   // CC.set_constant(0.0);
//   // {
//   //   timer.start();
//   //   DArray::DMatrixDescriptor<float> Ql {Q, 0, 0, n, qq*k};
//   //   DArray::DMatrixDescriptor<float> KQl {KQ, 0, 0, n, qq*k};
//   //   DArray::DMatrixDescriptor<float> CCl {CC, 0, 0, qq*k, qq*k};
//   //   DArray::matrix_multiply("T", "N", Ql, KQl, CCl, 1.0f, 0.0f);
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: C=QT*KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   // }
//   // g.barrier();
//   // // CC.collect_and_print("CC");

//   // DArray::DMatrix<float> U(g, qq*k, qq*k);
//   // U.set_constant(0.0);
//   // {
//   //   timer.start();
//   //   auto Ul=DArray::svd<float>(qq*k, CC, 0, 0, qq*k, qq*k);
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: SVD takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   //   U.dereplicate_in_all(Ul, 0, 0, qq*k, qq*k);
//   // }
//   // g.barrier();
//   // // U.collect_and_print("U");

//   // DArray::DMatrix<float> Z(g, n, k);
//   // Z.set_constant(0.0);
//   // {
//   //   timer.start();
//   //   auto Ql=Q.replicate_in_rows(0, 0, n, qq*k);
//   //   auto Ul=U.replicate_in_columns(0, 0, qq*k, k);
//   //   auto Zl=Z.local_view(0, 0, n, k);
//   //   // if(g.rank()==0) printf("Ql[%d,%d], Ul[%d,%d], Zl[%d,%d]\n", Ql.dims()[0], Ql.dims()[1], Ul.dims()[0], Ul.dims()[1], Zl.dims()[0], Zl.dims()[1]);
//   //   float one=1.0f, zero=0.0f;
//   //   sgemm_("N", "N", &Ql.dims()[0], &Ul.dims()[1], &Ql.dims()[1], &one, Ql.data(), &Ql.ld(), Ul.data(), &Ul.ld(), &zero, Zl.data(), &Zl.ld());
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: Creating Low Rank Approximator Z takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   // }
//   // g.barrier();
//   // Z.collect_and_print("Z");

//   // DArray::Knorm<float>(g.rank(), np, g, n, d, k, X, Y, Z, gamma, bs);
  
//   // DArray::DMatrix<double> Zd(g, n, k);{
//   //   for(int i=0; i<n; ++i){
//   //     for(int j=0; j<k; ++j){
//   //       Zd.at(i,j)=Z.at(i,j);
//   //     }
//   //   }
//   // }

//   cuda_finalize();  
//   MPI_Finalize();
// }












































// #include "darray.h"
// #include "read_input.h"
// #include "lapack_like.h"

// extern void cuda_init();
// extern void cuda_finalize();
// extern void LRA(int rank, int lm, int ln, int ld, int lk, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* A, int lda, float gamma, float* O, int ldo);
// extern void SGEQRF(int m, int n, float *Q, int ldq);
// extern void SORMQR(int m, int n, float *Q, int ldq, float *RQ, int ldrq);
// extern void TCGemm(int rank, int m, int n, int k, float* A, int lda, float* B, int ldb, float alpha, float beta, float* C, int ldc);
// extern void Chol(int k, float *C, int ldc);
// extern void transpose(float *dA, int lda, int m, int n);

// template<typename T>
// void QR(DArray::DMatrix<T> A, int i1, int j1, int i2, int j2) {
//   // 1. replicate in rows: rows distributed, col replicated
//   auto LA = A.replicate_in_rows(i1,j1,i2,j2);
//   int m = LA.dims()[0], n = LA.dims()[1];
//   assert( n == j2 - j1);
//   assert( m > n);
//   // 2. local QR (BLAS)
//   SGEQRF( m, n, LA.data(), LA.ld());
//   // 3. stack Rs into a big tall R, size (n*np) * n, e.g. 10,000 * 100
//   // FIXME: chnage datatype according to typename T.
//   int np = A.grid().dims()[0];
//   std::vector<T> recv(np*n*n);
//   {
//       DArray::LMatrix<T> R(n, n);
//       auto err = MPI_Allgather(R.data(), n * n, MPI_FLOAT, recv.data(), n * n, MPI_FLOAT, A.grid().comms()[0]);
//       assert(err == MPI_SUCCESS);
//   }
//   DArray::LMatrix<T> Rs(np*n, n);
//   int ldRs = Rs.ld();
//   for(int b=0; b<np; b++) {
//       T* recvblock = &recv.data()[n*n*b];
//       for(int j=0; j<n; j++) {
//           for (int i=0; i<n; i++)
//               Rs.data()[b*n+i+j*ldRs] = (j<i)? 0 : recvblock[i + j*n];
//       }
//   }
//   // 4. qr the stacked R
//   DArray::LMatrix<T> Q(np*n, n);
//   int ldq = Q.ld();
//   for (int i=0; i<ldq; i++) {
//       for (int j = 0; j < n; j++)
//           Q.data()[i + j * ldq] = (i == j) ? 1.0 : 0;
//   }
//   {
//     int m = np*n;
//     SGEQRF( m, n, Rs.data(), Rs.ld());
//     SORMQR(m, n, Rs.data(), Rs.ld(), Q.data(), Q.ld());
//   }
//   // 5. post-multiply the Q of stacked R
//   // LA Q has size m*n; R Q has size n*n
//   // LA Q * R Q -> m*n
//   // (I - 2*tau*H[1]*H[1]') m*m apply to n*n -> m*n
//   DArray::LMatrix<T> RQ(m, n);
//   auto pi = A.grid().ranks()[0];
//   int ldRQ = RQ.ld(), ldQ = Q.ld();
//   for(int j=0; j<n; j++) {
//     for(int i=0; i<n; i++) {
//         RQ.data()[i+j*ldRQ] = Q.data()[n*pi + i + j*ldQ];
//     }
//     for(int i=n; i<m; i++)
//         RQ.data()[i+j*ldRQ] = 0;
//   }
//   SORMQR(m, n, LA.data(), LA.ld(), RQ.data(), ldRQ);
//   // 6. write Q back to distributed A
//   auto grid = A.grid();
//   // grid.print(RQ, "RQ");
//   A.dereplicate_in_rows(RQ, i1, j1, i2, j2);
// }


// // // Distributed Low-Rank Approximated Kernel
// // void LowRankKernel(int rank, int m, int n, int d, DArray::LMatrix<float> Xi,  int ldxi, DArray::LMatrix<float> Xj,  int ldxj, 
// // DArray::LMatrix<float> Yi, DArray::LMatrix<float> Yj, float gamma){
// //   printf("Rank::%d, lm::%d, ln::%d, ld::%d, ldxi::%d, ldxj::%d\n", rank, m, n, d, ldxi, ldxj);
// // }


// int main(int argc, char** argv){
//   MPI_Init(&argc, &argv);
//   int np, p, q;
//   MPI_Comm_size(MPI_COMM_WORLD, &np);
//   DArray::Grid::np_to_pq(np, p, q); // np = p*q
//   DArray::Grid g(MPI_COMM_WORLD, p, q);
//   g.barrier();
//   cuda_init();
//   DArray::ElapsedTimer timer;

//   long long int gn=680;
// 	int d=10;
//   int k=16;
//  	int qq=1;
//   float gamma=0.0001f;
//   char *filename="/project/pwu/dataset/breast-cancer_scale";
//   // long long int gn=5000000;
// 	// int d=18;
// 	// int k=512;
// 	// int qq=1;
// 	// float gamma=0.0001f;
// 	// char *filename="/project/pwu/dataset/SUSY";


//   if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, gn::%d, d::%d, k::%d) \n", p, q, gn, d, k);
//   float *x=(float*)malloc(sizeof(float)*d*gn);
//   float *y=(float*)malloc(sizeof(float)*gn);
//   for(int i=0; i<d; i++){
//     for(int j=0; j<gn; j++){
//       x[i+j*d]=0.0;
//     }
//   }
//   for(int i=0; i<gn; ++i){
//     y[i]=0.0;
//   }
//   timer.start();
//   read_input_file(filename, gn, d, k, x, d, y);
//   int ms = timer.elapsed();
//   if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   g.barrier();

//   DArray::DMatrix<float> X(g, d, gn);
//   DArray::DMatrix<float> Y(g, 1, gn);
//   X.set_value_from_matrix(x, d);
//   Y.set_value_from_matrix(y, 1);

 
//   free(x);
//   free(y);
//   g.barrier();

//   timer.start();
//   DArray::LMatrix<float> Xi = X.transpose_and_replicate_in_rows(0, 0, d, gn);
//   DArray::LMatrix<float> Yi = Y.transpose_and_replicate_in_rows(0, 0, 1, gn);
//   DArray::LMatrix<float> X_rep = X.replicate_in_all(0, 0, d, gn);
//   DArray::LMatrix<float> Y_rep = Y.replicate_in_all(0, 0, 1, gn);
//   DArray::LMatrix<float> Xj = X_rep.transpose();
//   DArray::LMatrix<float> Yj = Y_rep.transpose();
//   ms = timer.elapsed();
//   if(g.rank()==0) fmt::print("P[{},{}]: Replicating and Distribution of rows and columns in Xi, Xj, Yi, Yj takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   int lm = Xi.dims()[0];
//   int ln = Xj.dims()[0];
//   int ld = Xi.dims()[1];

//   if(g.rank() == 0){
//     Xi.print("Xi");
//     Yi.print("Yi");
//     Xj.print("Xj");
//     Yj.print("Yj");
//   }
//   if(g.rank() == 0) printf("Rank::%d, Xi[%d,%d], Xj[%d,%d] \n Yi[%d,%d], Yj[%d,%d] \n", g.rank(), Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], Yi.dims()[0], 
//   Yi.dims()[1], Yj.dims()[0], Yj.dims()[1]);

//   DArray::DMatrix<float> A(g, gn, k);
//   {
//     DArray::DMatrix<float> Omega(g, gn, k);
//     Omega.set_normal_seed(0, 1, 1);
//     // Omega.collect_and_print("O");
//     DArray::LMatrix<float> Ol = Omega.replicate_in_columns(0, 0, gn, k);
//     int lk = Ol.dims()[1];
//     DArray::LMatrix<float> Al(gn, k);
//     Al.set_constant(0.0);
//     // if(g.rank() == 0) printf("Rank::%d, Ol[%d,%d], Al[%d,%d]", g.rank(), Ol.dims()[0], Ol.dims()[1], Al.dims()[0], Al.dims()[1]);
//     timer.start();
//     LRA(g.rank(), lm, ln, ld, lk, Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Yi.data(), Yj.data(), Al.data(), Al.ld(), gamma, Ol.data(), Ol.ld());
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//     A.dereplicate_in_all(Al, 0, 0, Al.dims()[0], Al.dims()[1]);
//   }
//   A.collect_and_print("A");
//   g.barrier();

//   // // Distributed QR
//   // DArray::DMatrix<float> Q = A.clone();
//   // {
//   //   timer.start();
//   //   QR(Q, 0, 0, gn, k);
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   // }
//   // g.barrier();

//   // DArray::DMatrix<float> KQ(g, gn, k);
//   // {
//   //   DArray::LMatrix<float> Ql = Q.replicate_in_columns(0, 0, gn, k);
//   //   int lk = Ql.dims()[1];
//   //   DArray::LMatrix<float> KQl = KQ.replicate_in_rows(0, 0, gn, k);
//   //   KQl.set_constant(0.0);
//   //   timer.start();
//   //   LRA(g.rank(), lm, ln, ld, lk, Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Yi.data(), Yj.data(), KQl.data(), KQl.ld(), gamma,  Ql.data(),  Ql.ld());
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   //   KQ.dereplicate_in_all(KQl, 0, 0, KQl.dims()[0], KQl.dims()[1]);
//   // }
//   //  g.barrier();
  
//   // DArray::DMatrix<float> C(g, k, k);
//   // {
//   //   DArray::LMatrix<float> Ql = Q.transpose_and_replicate_in_rows(0, 0, gn, k);
//   //   DArray::LMatrix<float> KQl = KQ.replicate_in_columns(0, 0, gn, k);
//   //   DArray::LMatrix<float> Cl = C.replicate_in_all(0, 0, k, k);
//   //   int ld=Ql.dims()[0];
//   //   int le=KQl.dims()[1];
//   //   int lf=Ql.dims()[1];
//   //   float sone=1.0f;
//   //   float szero=0.0f;
//   //   timer.start();
//   //   TCGemm(g.rank(), ld, le, lf, Ql.data(), Ql.ld(), KQl.data(), KQl.ld(), sone, szero, Cl.data(), Cl.ld());
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: TCGemm takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   //   if(g.rank() == 0) printf("Rank::%d, Ql[%d,%d], KQl[%d,%d], Cl[%d,%d]", g.rank(), Ql.dims()[0], Ql.dims()[1], KQl.dims()[0], KQl.dims()[1], Cl.dims()[0], Cl.dims()[1]);
//   //   C.dereplicate_in_all(Cl, 0, 0, Cl.dims()[0], Cl.dims()[1]);
//   // }
//   //  g.barrier();

//   cuda_finalize();
//   MPI_Finalize();
// }











//   // //Eigen Dicompose C = X*Σ*XT and generate Low Rank Approximator, U = Q*X*Σ1/2
//   // DArray::DMatrix<float> XX(g, k, k); 
//   // {
//   //   DArray::LMatrix<float> Cl = C.replicate_in_all(0, 0, k, k);
//   //   DArray::LMatrix<float> w(1,k);
//   //   DArray::LMatrix<float> Xl(k,k);
//   //   int lwork=-1, liwork=-1, info ;
//   //   float tmp, itmp;
//   //   ssyevd_("N", "L",  &k, Cl.data(), &Cl.ld(), w.data(), &tmp, &lwork, &itmp, &liwork, &info);
//   //   if(info!=0 && g.rank()==0) printf("Error: DSYEVD info=%d\n", info); 
		
// 	// 	if(g.rank()==0) printf("[LRA]: C: largest ew=%.3e, smallest ew=%.3e\n", w[k-1], w[0]);
//   //   int i, realk=k; 
// 	// 	// C = Xl*Xl'
// 	// 	for (i=0; i<k; i++) {
// 	// 		double s; 
// 	// 		if (w[i] > 0 ) {
// 	// 			s = sqrt(w[i]); 
// 	// 		} else {
// 	// 			s = 0;
// 	// 			realk--;
// 	// 		}
// 	// 		for (int j=0; j<k; j++) {
// 	// 			Xl[j+i*k] = s * Cl[j+i*k]; 
// 	// 		}
// 	// 	}
// 	// 	if(g.rank()==0) printf("real rank is %d\n", realk); 
//   //   XX.dereplicate_in_all(Xl, 0, 0, k, k);
//   // }

//   // // DArray::DMatrix<float> U(g, gn, k);
//   // // {
//   // //   DArray::LMatrix<float> Ql = Q.replicate_in_rows(0, 0, gn, k);
//   // //   DArray::LMatrix<float> Xl = XX.replicate_in_columns(0, 0, k, k);
//   // //   DArray::LMatrix<float> tmp(gn, k);
//   // //   // if(g.rank()==0) printf("Ql[%d,%d], Xl[%d,%d] tmp[%d,%d]\n", Ql.dims()[0], Ql.dims()[1],  Xl.dims()[0],  Xl.dims()[1], tmp.dims()[0], tmp.dims()[1]);
//   // //   TCGemm(Ql.dims()[0], Xl.dims()[1], Ql.dims()[1], Ql.data(), Ql.ld(), Xl.data(), Xl.ld(), 1.0, 0.0, tmp.data(), tmp.ld());
//   // //   U.dereplicate_in_all(tmp, 0, 0, tmp.dims()[0], tmp.dims()[1]);
//   // // }




//   // if(g.rank() == 0) printf("Rank::%d, lm::%d, ln::%d, ld::%d \nXii[%d,%d], XjjT[%d,%d] Yii[%d,%d], YjjT[%d,%d]\n", g.rank(), lm, ln, ld, Xii.dims()[0], Xii.dims()[1], XjjT.dims()[0], XjjT.dims()[1], Yii.dims()[0], 
//   // Yii.dims()[1], YjjT.dims()[0], YjjT.dims()[1]);

//   // {
//   //   for(int r=0; r<np; r++){
//   //     int ie=(lm*(r+1))/p;
//   //     int is=(lm*r)/p;
//   //     int je=(ln*(r+1))/p;
//   //     int js=(ln*r)/p;
//   //     int NNi = ie-is;
//   //     int NNj = je-js;
//   //     LowRankKernel(g.rank(), lm, ln, ld, Xii, Xii.ld(), XjjT, XjjT.ld(), Yii, YjjT, gamma);
//   //   }
//   // }
