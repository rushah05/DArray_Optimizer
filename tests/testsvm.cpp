#include "darray.h"
#include "read_input.h"
#include "lapack_like.h"
#include "cblas_like.h"
#include "dkernel.h"
#include "dbarrier.h"

float gaussrand(){
    static float V1, V2, S;
    static int phase = 0;
    float X;
    
    if(phase == 0) {
        do {
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;
            
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
  // cuda_init();
  DArray::ElapsedTimer timer;

  if(argc < 9){
      if(g.rank() == 0) printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(double)> <power refinement q(int)>\n");
      return 0;
  }

  char *filename = argv[1];
  // long long int n = atoi(argv[2]);
  int n = atoi(argv[2]);
  int d = atoi(argv[3]);
  int k = atoi(argv[4]); /*no of columns of the tall matrix A, B and Omega*/
  int bs = atoi(argv[5]); 
  double gamma = strtod(argv[6], NULL);
  int qq = atoi(argv[7]);
  double C = strtod(argv[8], NULL);

  if(qq < 1) qq=1;

  if(bs <= 0) bs = 128;

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
  
  if(g.rank() == 0) {
    timer.start();
    read_input_file(g.rank(), filename, n, d, x, d, y);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  MPI_Bcast(x, d*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  g.barrier();

  DArray::DMatrix<float> X(g, d, n);
  DArray::DMatrix<float> Y(g, 1, n);
  X.set_value_from_matrix(x, d);
  Y.set_value_from_matrix(y, 1);
  free(x);
  free(y);
  g.barrier();


  if(g.rank()==0) fmt::print("Starting LRA (low-rank approximation of kernel) ... K({},{}) ~ G({},{})\n", n, n, n, k);
  DArray::DMatrix<float> A(g, n, k);
  A.set_constant(0.0);
  {
    DArray::DMatrix<float> O(g, n, k);
    for(int i=0; i<n; i++){
      for(int j=0; j<k; j++){
        O.at(i,j)=gaussrand();
      }
    } 

    DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, O, A, gamma, bs);
  }
  // A.collect_and_print("A");
  g.barrier();


  if(g.rank()==0) fmt::print("Creating a Krylov subspace... ... ({}, {}) \n", n, qq*k);
  DArray::DMatrix<float> K(g, n, qq*k);
  K.set_constant(0.0);
  {
    DArray::DMatrixDescriptor<float> Kdesc{K, 0, 0, n, qq*k}, Adesc{A, 0, 0, n, k};
    auto Kl=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j2);
    auto Al=Adesc.matrix.local_view(Adesc.i1, Adesc.j1, Adesc.i2, Adesc.j2);
    DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), Kl.data(), Kl.ld());
    
    for(int i=2; i<=qq; ++i){
      auto O=A.clone();
      A.set_constant(0);
      DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, O, A, gamma, bs);
      auto Kl=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1+(i-1)*k, Kdesc.i2, Kdesc.j1+(i*k));
      auto Al=Adesc.matrix.local_view(Adesc.i1, Adesc.j1, Adesc.i2, Adesc.j2);
      if(g.rank() == 0) printf("Al[%d,%d], Kl[%d,%d] - (%d,%d,%d,%d) \n", Al.dims()[0], Al.dims()[1], Kl.dims()[0], Kl.dims()[1], Kdesc.i1, Kdesc.j1+(i-1)*k, Kdesc.i2, Kdesc.j1+(i*k));
      // DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*k*Kl.dims()[0]], Kl.ld());
    }
  }
  // K.collect_and_print("K");
  g.barrier();


  // if(g.rank()==0) fmt::print("Starting LRA (low-rank approximation of kernel) and Creating a Krylov subspace... ...\n");
  // DArray::DMatrix<float> K(g, n, qq*k);
  // {
  //   DArray::DMatrix<float> A(g, n, k);
  //   A.set_constant(0.0);
  //   DArray::DMatrix<float> O(g, n, k);
  //   for(int i=0; i<n; i++){
  //     for(int j=0; j<k; j++){
  //       O.at(i,j)=gaussrand();
  //     }
  //   } 

  //   timer.start();
  //   for(int i=1; i<=qq; ++i){
  //     A.set_constant(0.0);
  //     DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, O, A, gamma);
  //     auto Kl=K.local_view(0, 0, n, qq*k);
  //     auto Al=A.local_view(0, 0, n, k);
  //     // if(g.rank()==0) printf("Kl[%d,%d], Al[%d,%d], K[%d,%d] A[%d,%d] \n", Kl.dims()[0], Kl.dims()[1], Al.dims()[0], Al.dims()[1],
  //     // K.dims()[0], K.dims()[1], A.dims()[0], A.dims()[1]);
  //     int lk=Al.dims()[1];
  //     DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*lk*Kl.ld()], Kl.ld());
  //     O=A.clone();
  //   }
  //   int ms = timer.elapsed();
  //   if(g.rank()==0) fmt::print("P[{},{}]: LRA with Blocked Lanczos takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // }
  // g.barrier();

  // K.collect_and_print("K");

  // // Distributed QR
  // DArray::DMatrix<float> Q = K.clone();
  // {
  //   timer.start();
  //   tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
  //   int ms = timer.elapsed();
  //   if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // }
  // g.barrier();


  // // Distributed KQ
  // DArray::DMatrix<float> KQ(g, n, qq*k);
  // {
  //   timer.start();
  //   DArray::LRA<float>(g.rank(), np, g, n, d, qq*k, X, Y, Q, KQ, gamma);
  //   int ms = timer.elapsed();
  //   if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // }
  // g.barrier();


  // DArray::DMatrix<float> CC(g, qq*k, qq*k);
  // {
  //   timer.start();
  //   DArray::DMatrixDescriptor<float> Ql {Q, 0, 0, n, qq*k};
  //   DArray::DMatrixDescriptor<float> KQl {KQ, 0, 0, n, qq*k};
  //   DArray::DMatrixDescriptor<float> CCl {CC, 0, 0, qq*k, qq*k};
  //   DArray::matrix_multiply("T", "N", Ql, KQl, CCl, 1.0f, 0.0f);
  //   int ms = timer.elapsed();
  //   if(g.rank()==0) fmt::print("P[{},{}]: C=QT*KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // }
  // g.barrier();
  // // CC.collect_and_print("CC");

  // DArray::DMatrix<float> U(g, qq*k, qq*k);
  // {
  //   timer.start();
  //   auto Ul=DArray::svd(qq*k, CC, 0, 0, qq*k, qq*k);
  //   int ms = timer.elapsed();
  //   if(g.rank()==0) fmt::print("P[{},{}]: SVD takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  //   U.dereplicate_in_all(Ul, 0, 0, qq*k, qq*k);
  // }
  // g.barrier();

  // DArray::DMatrix<float> Z(g, n, k);
  // {
  //   timer.start();
  //   DArray::DMatrixDescriptor<float> Ql {Q, 0, 0, n, qq*k};
  //   DArray::DMatrixDescriptor<float> Ul {U, 0, 0, qq*k, k};
  //   DArray::DMatrixDescriptor<float> Zl {Z, 0, 0, n, k};
  //   // DArray::matrix_multiply("N", "N", Ql, Ul, Zl, 1.0f, 0.0f);
  //   int ms = timer.elapsed();
  //   if(g.rank()==0) fmt::print("P[{},{}]: Creating Low Rank Approximator Z takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // }
  // g.barrier();

  // DArray::Knorm<float>(g.rank(), np, g, n, d, k, X, Y, Z, gamma);
      
  MPI_Finalize();
}



































// // Distributed Low Rank Approximation of Kernel
//   DArray::DMatrix<float> A(g, n, k);
//   A.set_constant(0.0);
//   {
//     DArray::DMatrix<float> O(g, n, k);
//     O.set_normal_seed(0, 1, 1);
//     timer.start();
//     DArray::LRA<float>(g.rank(), p, g, n, d, k, X, Y, O, A, gamma);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   // A.collect_and_print("A");


//   if(g.rank()==0) fmt::print("\n");
//   // Creating a Krylov subsbace
//   DArray::DMatrix<float> K(g, n, qq*k);
//   K.set_constant(0.0);
//   {
//     DArray::copy_block<float>(A.ldims()[0], A.ldims()[1], A.data(), A.lld(), K.data(), K.lld());
//     for(int i=1; i<qq; ++i){
//       auto O=A.clone();
//       A.set_constant(0.0);
//       DArray::LRA<float>(g.rank(), p, g, n, d, k, X, Y, O, A, gamma);
//       auto Kl=K.local_view(0, 0, n, qq*k);
//       auto Al=A.local_view(0, 0, n, k);
//       DArray::copy_block<float>(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*lk*Kl.ld()], Kl.ld());
//     }  
//   }





//  // Distributed Low Rank Approximation of Kernel and creating a Krylov subsbace
//   DArray::DMatrix<float> K(g, n, qq*k);
//   {
//     DArray::DMatrix<float> A(g, n, k);
//     DArray::DMatrix<float> O(g, n, k);
//     O.set_normal_seed(0, 1, 1);
    
//     timer.start();
//     for(int i=1; i<=qq; ++i){
//       DArray::LRA<float>(g.rank(), np, n, d, k, X, Y, O, A, gamma);
//       auto Kl=K.local_view(0, 0, n, qq*k);
//       auto Al=A.local_view(0, 0, n, k);
//       // if(g.rank()==0) printf("Kl[%d,%d], Al[%d,%d], K[%d,%d] A[%d,%d] \n", Kl.dims()[0], Kl.dims()[1], Al.dims()[0], Al.dims()[1],
//       // K.dims()[0], K.dims()[1], A.dims()[0], A.dims()[1]);
//       int lk=Al.dims()[1];
//       DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*lk*Kl.ld()], Kl.ld());
//       O=A.clone();
//     }
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }

//   // Distributed QR
//   DArray::DMatrix<float> Q = K.clone();
//   {
//     timer.start();
//     tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();


//   // Distributed KQ
//   DArray::DMatrix<float> KQ(g, n, qq*k);
//   {
//     timer.start();
//     DArray::LRA<float>(g.rank(), np, n, d, qq*k, X, Y, Q, KQ, gamma);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();


//   DArray::DMatrix<float> CC(g, qq*k, qq*k);
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
//   {
//     timer.start();
//     auto Ul=DArray::svd(qq*k, CC, 0, 0, qq*k, qq*k);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: SVD takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//     U.dereplicate_in_all(Ul, 0, 0, qq*k, qq*k);
//   }
//   g.barrier();

//   DArray::DMatrix<float> Z(g, n, k);
//   {
//     timer.start();
//     DArray::DMatrixDescriptor<float> Ql {Q, 0, 0, n, qq*k};
//     DArray::DMatrixDescriptor<float> Ul {U, 0, 0, qq*k, k};
//     DArray::DMatrixDescriptor<float> Zl {Z, 0, 0, n, k};
//     // DArray::matrix_multiply("N", "N", Ql, Ul, Zl, 1.0f, 0.0f);
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: Creating Low Rank Approximator Z takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();

//   DArray::Knorm<float>(g.rank(), np, g, n, d, k, X, Y, Z, gamma);































// // Distributed Low Rank Approximation of Kernel and creating a Krylov subsbace
//   DArray::DMatrix<float> K(g, n, qq*k);
//   {
//     DArray::DMatrix<float> A(g, n, k);
//     DArray::DMatrix<float> O(g, n, k);
//     O.set_normal_seed(0, 1, 1);

//     timer.start();
//     for(int i=1; i<=qq; ++i){
//       DArray::LRA<float>(g.rank(), np, n, d, k, X, Y, O, A, gamma);
//       auto Kl=K.local_view(0, 0, n, qq*k);
//       auto Al=A.local_view(0, 0, n, k);
//       int lk=Al.dims()[1];
//       DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*lk*Kl.ld()], Kl.ld());
//       O=A.clone();
//     }
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: LRA : {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }


// // Distributed Low Rank Approximation of Kernel and creating a Krylov subsbace
//   DArray::DMatrix<float> K(g, n, qq*k);
//   {
//     DArray::DMatrix<float> A(g, n, k);
//     DArray::DMatrix<float> O(g, n, k);
//     O.set_normal_seed(0, 1, 1);
    
//     timer.start();
//     for(int i=1; i<=qq; ++i){
//       DArray::LRA<float>(g.rank(), np, n, d, k, X, Y, O, A, gamma);
//       auto Kl=K.local_view(0, 0, n, qq*k);
//       auto Al=A.local_view(0, 0, n, k);
//       int lk=Al.dims()[1];
//       DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*lk*Kl.ld()], Kl.ld());
//       O=A.clone();
//     }
//     int ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: LRA : {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }

//   // // Distributed QR
//   // DArray::DMatrix<float> Q = K.clone();
//   // {
//   //   timer.start();
//   //   tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   // }
//   // g.barrier();


//   // // Distributed KQ
//   // DArray::DMatrix<float> KQ(g, n, qq*k);
//   // {
//   //   timer.start();
//   //   DArray::LRA<float>(g.rank(), np, n, d, qq*k, X, Y, Q, KQ, gamma);
//   //   int ms = timer.elapsed();
//   //   if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   // }
//   // g.barrier();


//   // DArray::DMatrix<float> CC(g, qq*k, qq*k);
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





















// timer.start();
//   DArray::DMatrix<float> K(g, n, qq*k);
//   {
//     DArray::DMatrix<float> A(g, n, k);
//     DArray::DMatrix<float> O(g, n, k);
//     O.set_normal_seed(0, 1, 1);

//     for(int i=1; i<=qq; ++i){
//       DArray::LRA<float>(g.rank(), np, n, d, k, X, Y, O, A, gamma);
//       auto Kl=K.local_view(0, 0, n, qq*k);
//       auto Al=A.local_view(0, 0, n, k);
//       int lk=Al.dims()[1];
//       // if(g.rank()==0) printf("Kl[%d,%d] Al[%d,%d] \n", Kl.dims()[0], Kl.dims()[1], Al.dims()[0], Al.dims()[1]);
//       DArray::copy_block(Al.dims()[0], Al.dims()[1], Al.data(), Al.ld(), &Kl.data()[(i-1)*lk*Kl.ld()], Kl.ld());
//     }
    
//     if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   // K.collect_and_print("K");
//   // // auto Kl=K.collect(0);
//   // // if(rk == 0) printMatrix("K.csv", n, qq*k, Kl, Kl.ld());
//   g.barrier();

//   // Distributed QR
//   DArray::DMatrix<float> Q = K.clone();
//   {
//     timer.start();
//     tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
//     ms = timer.elapsed();
//     if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
//   }
//   g.barrier();







  // // auto Xl = X.collect(0);
  // // auto Yl = Y.collect(0);
  // // if(g.rank() ==0){
  // //   printMatrix("X.csv", d, gn, Xl, Xl.ld());
  // //   printMatrix("Y.csv", 1, gn, Yl, Yl.ld());
  // // }



  // K.set_constant(0.0);
  // {
  //   DArray::DMatrix<float> A(g, gn, k);
  //   DArray::DMatrix<float> O(g, gn, k);
  //   O.set_normal_seed(0, 1, 1);
  //   // auto Ol = O.collect(0);
  //   // if(g.rank() ==0){
  //   //   printMatrix("O.csv", gn, k, Ol, Ol.ld());
  //   // }
  //   timer.start();
  //   DArray::LRA<float>(g.rank(), np, gn, d, k, X, Y, O, A, gamma);
  //   // auto Kl=K.local_view(0, 0, gn, k);
  //   // auto Al=A.local_view(0, 0, gn, k);
  //   // DArray::copy_block(Kl.dims()[0], Kl.dims()[1], &Al.data()[0], Al.ld(), &Kl.data()[0], Kl.ld());

  //   // for(int i=1; i<qq; ++i){
  //   //   DArray::DMatrix<float> KA(g, gn, k);
  //   `   gn, d, k, X, Y, A, KA, gamma);
  //   //   auto Kl=K.local_view(0, i*k, gn, (i+1)*k);
  //   //   auto KAl=KA.local_view(0, 0, gn, k);
  //   //   // if(g.rank()==0) printf("K[%d,%d], KA[%d,%d], KAl[%d,%d], Kl[%d,%d]\n", K.dims()[0], K.dims()[1], KA.dims()[0], KA.dims()[1], KAl.dims()[0], KAl.dims()[1], Kl.dims()[0], Kl.dims()[1]);
  //   //   DArray::copy_block(KAl.dims()[0], KAl.dims()[1], &KAl.data()[0], KAl.ld(), &Kl.data()[0], Kl.ld());
  //   //   A=KA;
  //   // } 
  //   int ms = timer.elapsed();
  //   if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // }
  // g.barrier();
  // // auto Kl = K.collect(0);
  // // if(g.rank() ==0){
  // //     printMatrix("K.csv", gn, qq*k, Kl, Kl.ld());
  // // }


 

  // // // DArray::DMatrix<float> CC(g, qq*k, qq*k);
  // // // {
  // // //   timer.start();
  // // //   auto Ql = Q.replicate_in_rows(0, 0, gn, qq*k);
  // // //   auto KQl = KQ.replicate_in_rows(0, 0, gn, qq*k);
  // // //   DArray::LMatrix<float> QKQ (qq*k, qq*k);
  // // //   int ln=Ql.dims()[0];
  // // //   int lk=Ql.dims()[1];
  // // //   float sone=1.0f, szero=0.0f;
  // // //   sgemm_("T", "N", &lk, &lk, &ln, &sone, Ql.data(), &Ql.ld(), KQl.data(), &KQl.ld(), &szero, QKQ.data(), &QKQ.ld()); 
  // // //   CC.dereplicate_in_all(QKQ, 0, 0, qq*k, qq*k);
  // // //   int ms = timer.elapsed();
  // // //   if(g.rank()==0) fmt::print("P[{},{}]: C=QT*KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // // // }
  // // // g.barrier();
  
  // // // DArray::DMatrix<float> U(g, qq*k, qq*k);
  // // // {
  // // //   timer.start();
  // // //   auto Ul=DArray::svd(qq*k, CC, 0, 0, qq*k, qq*k);
  // // //   int ms = timer.elapsed();
  // // //   if(g.rank()==0) fmt::print("P[{},{}]: SVD takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // // //   U.dereplicate_in_all(Ul, 0, 0, Ul.dims()[0], Ul.dims()[1]);
  // // // }

  // // // DArray::DMatrix<float> Z(g, gn, k);
  // // // {
  // // //   timer.start();
  // // //   auto Ql = Q.replicate_in_rows(0, 0, gn, qq*k);
  // // //   auto Ul = U.replicate_in_columns(0, 0, qq*k, k);
  // // //   int lm=Ql.dims()[0];
  // // //   int ln=Ql.dims()[1];
  // // //   int lk=Ul.dims()[1];
  // // //   DArray::LMatrix<float> Zl(gn, k);
  // // //   float sone=1.0f, szero=0.0f;
  // // //   // printf("Ql[%d,%d], Ul[%d,%d], Zl[%d,%d]\n", Ql.dims()[0], Ql.dims()[1], Ul.dims()[0], Ul.dims()[1], Zl.dims()[0], Zl.dims()[1]);
  // // //   sgemm_("N", "N", &lm, &lk, &ln, &sone, Ql.data(), &Ql.ld(), Ul.data(), &Ul.ld(), &szero, Zl.data(), &Zl.ld());
  // // //   int ms = timer.elapsed();
  // // //   // if(g.rank()==0) fmt::print("P[{},{}]: Creating Low Rank Approximator Z takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // // //   Z.dereplicate_in_all(Zl, 0, 0, Zl.dims()[0], Zl.dims()[1]);
  // // // }

   
  // // // auto Zl = Z.collect(0);
  // // // if(g.rank() ==0){
  // // //   printMatrix("Zl.csv", gn, k, Zl, Zl.ld());
  // // // }
  // // // DArray::K_QCQ_norm<float>(g.rank(), np, gn, d, k, X, Y, Q, gamma);


  // // // // PHASE 2
  // // // {
  // // //   DArray::LMatrix<float> a(1, gn);
  // // //   auto Zl = Z.replicate_in_all(0, 0, gn, k);
  // // //   auto Yl = Y.replicate_in_all(0, 0, 1, gn);
  // // //   int plus=0, minus=0;

  // // //   for(int i=0; i<gn; i++){
  // // //     if(Yl.data()[i]==1) plus+=1;
  // // //     else if(Yl.data()[i]==-1) minus+=1;
  // // //   }


  // // //   // determine the initial values according to Y[i]==+-1
  // // //   float mval=0.0, pval=0.0;
  // // //   if(plus > minus){
  // // //       mval = 0.9*C;
  // // //       pval = mval*minus/plus;
  // // //   }else { 
  // // //       pval = 0.9*C;
  // // //       mval = pval*plus/minus;
  // // //   }

  // // //   if(g.rank()==0) printf("plus::%d, minus::%d, mval::%f, pval::%f\n", plus, minus, mval, pval);

  // // //   for(int i=0; i<gn; ++i){
  // // //       if(Yl.data()[i]==1) {
  // // //         a.data()[i] = pval;
  // // //       }else if (Yl.data()[i] == -1){
  // // //         a.data()[i] = mval;
  // // //       }else{
  // // //         assert(false);
  // // //       }
  // // //   }
 
  // // //   // feasibility:
  // // //   for(int i=0; i<gn; ++i){
  // // //     if(a.data()[i]<0.0 || a.data()[i]>C){
  // // //       printf("[Rank::%d] Initial guess Inequality violation! offending a[i]: i::%d, a[i]::%f)", g.rank(), i, a.data()[i]);
  // // //       break;
  // // //     }
  // // //   }


  // // //   int n=(int)gn; 
  // // //   float Yta = sdot_(&n, Yl.data(), &Yl.ld(), a.data(), &a.ld());
  // // //   if(g.rank() == 0) printf("Initial feasibility: Y'*a=%f\n", Yta);

  // // //   int t = 1;
  // // //   int mu = 20;
  // // //   float alpha = 0.1;
  // // //   float beta = 0.5;

  // // //   // barrier outer iteration
  // // //   DArray::LMatrix<float> D(n, 1);
  // // //   int bi=1; //barrier iteration

  // // //   // Z.collect_and_print("Z");
  // // //   // if(g.rank() == 0) a.print("a");
  // // //   float fsum = abs(f(g.rank(), n, k, a, Zl));

  // // //   // while(true) {
  // // //   //   // println("Barrier iteration #$(bi); suboptimality gap n/t=$(n/t), n=$(n) t=$(t), f(a) = $(f(a))")

  // // //   //   if (2*n/t < abs(f(n, k, a, Zl)) * 1.0e-5){ // f(a) need to converge to 0.1% around f*
  // // //   //     println("Barrier converged in $(bi) iterations; suboptimality gap 2n/t=$(2*n/t), n,t=$(n),$(t)")
  // // //   //     break;
  // // //   //   }
      
  // // //   //   t=mu*t;
  // // //   //   bi+=1;
  // // //   // }

  // // // }