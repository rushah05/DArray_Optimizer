#include "darray.h"
#include "read_input.h"
#include "lapack_like.h"
#include "dkernel.h"


int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int np, p, q;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  DArray::Grid::np_to_pq(np, p, q); // np = p*q
  DArray::Grid g(MPI_COMM_WORLD, p, q);
  g.barrier();
  // cuda_init();
  DArray::ElapsedTimer timer;

  if(argc < 8){
      if(g.rank() == 0) printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(double)> <power refinement q(int)>\n");
      return 0;
  }

  char *filename = argv[1];
  long long int gn = atoi(argv[2]);
  int d = atoi(argv[3]);
  int k = atoi(argv[4]); /*no of columns of the tall matrix A, B and Omega*/
  double gamma = strtod(argv[5], NULL);
  int qq = atoi(argv[6]);
  double C = strtod(argv[7], NULL);

  if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, gn::%d, d::%d, k::%d) \n", p, q, gn, d, k);
  float *x=(float*)malloc(sizeof(float)*d*gn);
  float *y=(float*)malloc(sizeof(float)*gn);
  for(int i=0; i<d; i++){
    for(int j=0; j<gn; j++){
      x[i+j*d]=0.0;
    }
  }
  for(int i=0; i<gn; ++i){
    y[i]=0.0;
  }
  timer.start();
  // if(g.rank() == 0){ 
  read_input_file(g.rank(), filename, gn, d, x, d, y);
  // }
  int ms = timer.elapsed();
  if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  // MPI_Bcast(x, (d*gn), DArray::convert_to_mpi_datatype<float>(), 0, g.comm());
  // MPI_Bcast(y, (gn), DArray::convert_to_mpi_datatype<float>(), 0, g.comm()); 
  g.barrier();

  DArray::DMatrix<float> X(g, d, gn);
  DArray::DMatrix<float> Y(g, 1, gn);
  X.set_value_from_matrix(x, d);
  Y.set_value_from_matrix(y, 1);
  free(x);
  free(y);
  g.barrier();


  // Distributed Low Rank Approximation of Kernel and creating a Krylov subsbace
  DArray::DMatrix<float> K(g, gn, qq*k);
  K.set_constant(0.0);
  {
    DArray::DMatrix<float> A(g, gn, k);
    DArray::DMatrix<float> O(g, gn, k);
    O.set_normal_seed(0, 1, 1);
    timer.start();
    DArray::LRA<float>(g.rank(), np, gn, d, k, X, Y, O, A, gamma);
    auto Kl=K.local_view(0, 0, gn, k);
    auto Al=A.local_view(0, 0, gn, k);
    DArray::copy_block(Kl.dims()[0], Kl.dims()[1], &Al.data()[0], Al.ld(), &Kl.data()[0], Kl.ld());

    for(int i=1; i<qq; ++i){
      DArray::DMatrix<float> KA(g, gn, k);
      KA.set_constant(0.0);
      DArray::LRA<float>(g.rank(), np, gn, d, k, X, Y, A, KA, gamma);
      auto Kl=K.local_view(0, i*k, gn, (i+1)*k);
      auto KAl=KA.local_view(0, 0, gn, k);
      // if(g.rank()==0) printf("K[%d,%d], KA[%d,%d], KAl[%d,%d], Kl[%d,%d]\n", K.dims()[0], K.dims()[1], KA.dims()[0], KA.dims()[1], KAl.dims()[0], KAl.dims()[1], Kl.dims()[0], Kl.dims()[1]);
      DArray::copy_block(KAl.dims()[0], KAl.dims()[1], &KAl.data()[0], KAl.ld(), &Kl.data()[0], Kl.ld());
      A=KA;
    } 
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  // K.collect_and_print("K");
  g.barrier();


  // Distributed QR
  DArray::DMatrix<float> Q = K.clone();
  {
    timer.start();
    tall_skinny_qr_factorize(Q, 0, 0, gn, qq*k);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  g.barrier();

// Ruchi ---- KQ is the problem
  DArray::DMatrix<float> KQ(g, gn, qq*k);
  {
    timer.start();
    DArray::LRA<float>(g.rank(), np, gn, d, qq*k, X, Y, Q, KQ, gamma);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }


  DArray::DMatrix<float> CC(g, qq*k, qq*k);
  {
    timer.start();
    auto Ql = Q.replicate_in_rows(0, 0, gn, qq*k);
    auto KQl = KQ.replicate_in_rows(0, 0, gn, qq*k);
    DArray::LMatrix<float> QKQ (qq*k, qq*k);
    int ln=Ql.dims()[0];
    int lk=Ql.dims()[1];
    float sone=1.0f, szero=0.0f;
    sgemm_("T", "N", &lk, &lk, &ln, &sone, Ql.data(), &Ql.ld(), KQl.data(), &KQl.ld(), &szero, QKQ.data(), &QKQ.ld()); 
    CC.dereplicate_in_all(QKQ, 0, 0, qq*k, qq*k);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: C=QT*KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  // CC.collect_and_print("C after multiplication");

  DArray::DMatrix<float> E(g, k, k);
  {
    auto El=DArray::svd(k, CC, 0, 0, k, k);
    if(g.rank() == 0) El.print("El");
    // E.dereplicate_in_all(0, 0, k, k);
  }

  MPI_Finalize();
}







