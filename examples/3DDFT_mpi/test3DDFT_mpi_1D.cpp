#include <mpi.h>
#include <complex>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include "device_macros.h"

#include "fftx_mpi.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  int p;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc != 7) {
    printf("usage: %s <M> <N> <K> <batch> <embedded> <forward>\n", argv[0]);
    exit(-1);
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int batch = atoi(argv[4]);
  bool is_embedded = 0 < atoi(argv[5]);
  bool is_forward = 0 < atoi(argv[6]);

  bool is_complex = false;
  bool R2C = !is_complex &&  is_forward;
  bool C2R = !is_complex && !is_forward;
  bool C2C =  is_complex;
  // X dim is size M, Y dim is size N, Z dim is size K.
  // R2C input is [K,       N, M]         doubles, block distributed Z.
  // C2R input is [N, M/2 + 1, K] complex doubles, block distributed X.
  // C2C input is [K,       N, M] complex doubles, block distributed Z.
  int Mi, Ni, Ki;
  int Mo, No, Ko;
  Mi = C2R ? (M/2) + 1 : M;
  Ni = N;
  Ki = K;
  Mo = M * (is_embedded ? 2 : 1); // TODO: change for R2C?
  No = N * (is_embedded ? 2 : 1);
  Ko = K * (is_embedded ? 2 : 1);

  double          *host_in , *dev_in;
  complex<double> *host_out, *dev_out, *Q3;

  // TODO: update C2R test to assume X is distributed initially.

  // X is first dim, so embed in dim of size Mo.
  if (is_forward) {
    host_in  = (double *         ) malloc(sizeof(complex<double>) * (Ki/p) * Ni * Mo * batch);
    host_out = (complex<double> *) malloc(sizeof(complex<double>) * (Ko/p) * No * Mo * batch);

    DEVICE_MALLOC(&dev_in , sizeof(complex<double>) * (Ki/p) * Ni * Mo * batch);
    DEVICE_MALLOC(&dev_out, sizeof(complex<double>) * (Ko/p) * No * Mo * batch);

    for (int l = 0; l < Ki/p; l++) {
      for (int j = 0; j < Ni; j++) {
        for (int i = 0; i < Mo; i++) {
          for (int b = 0; b < batch; b++) {
            host_in[(l * Ni*Mo + j * Mo + i)*batch + b] = (
              is_embedded && (i < Mi/2 || 3 * Mi/2 <= i) ||
              !is_forward
              ) ?
                0.0:
                1.0 * (b + 1) * (rank*0 + 1) * (i*0 + 1) * (j*0 + 1) * (l*0 + 1);
            // complex<double>(l*M*N*K  + k+1.0, 0.0);
          }
        }
      }
    }

    DEVICE_MEM_COPY(dev_in, host_in, sizeof(double) * (Ki/p) * Ni * Mo * batch, MEM_COPY_HOST_TO_DEVICE);
  } else {
    // TODO: fix for embedded.
    int M0 = Mi / p;
    if (M0*p < Mi) {
      M0 += 1;
    }
    int M1 = p;
    host_in  = (double *         ) malloc(sizeof(complex<double>) * Ki * Ni * M0 * batch);
    host_out = (complex<double> *) malloc(sizeof(complex<double>) * Ko * No * M0 * batch);

    DEVICE_MALLOC(&dev_in , sizeof(complex<double>) * Ki * Ni * M0 * batch);
    DEVICE_MALLOC(&dev_out, sizeof(complex<double>) * Ko * No * M0 * batch);

    complex<double> *in = (complex<double> *) host_in; // cast input pointer to complex for C2R

    // assume layout is [Y, X'/px, Z] (slowest to fastest)
    for (int j = 0; j < Ni; j++) {
      for (int i = 0; i < M0; i++) {
        for (int l = 0; l < Ki; l++) {
          for (int b = 0; b < batch; b++) {
            in[(j * M0*Ki + i * Ki + l)*batch + b] = {};
          }
        }
      }
    }
    if (rank == 0) {
      for (int b = 0; b < batch; b++) {
        in[b].real((M * N * K * (b+1)));
        in[b].imag(0);
      }
    }
    DEVICE_MEM_COPY(dev_in, host_in, sizeof(complex<double>) * Ki * Ni * M0 * batch, MEM_COPY_HOST_TO_DEVICE);
  } // end forward/inverse check.

  // TODO: resume conversion of forward to inverse from here.


  fftx_plan plan = fftx_plan_distributed_1d(p, M, N, K, 1, is_embedded, is_complex);
  for (int t = 0; t < 10; t++) {
    double start_time = MPI_Wtime();
    for (int b = 0; b < batch; b++) {
      fftx_execute_1d(plan, (double*)dev_out + b, dev_in + b, (is_forward ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE));
    }
    double end_time = MPI_Wtime();
    double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);
    if (rank == 0) {
      cout << M << "," << N << "," << K  << "," << batch  << "," << is_embedded << "," << is_forward << "," << max_time << endl;
    }
  }


  // fftx_plan plan = fftx_plan_distributed_1d(p, M, N, K, batch, is_embedded, is_complex);

  // for (int t = 0; t < 1; t++) {
  //   double start_time = MPI_Wtime();
  //   fftx_execute_1d(plan, (double*)dev_out, dev_in, (is_forward ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE));
  //   double end_time = MPI_Wtime();

  //   double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);

  // // // layout is [Y, X'/px, Z]
  // // DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);
  // // double diff = ((double) M * N * K) - host_out[0].real();
  //   if (rank == 0) {
  //     // cout << "end_to_end," << max_time<<endl;
  //     cout << M << "," << N << "," << K  << "," << batch  << "," << is_embedded << "," << is_forward << "," << max_time << endl;
  //     // cout << M << "," << N << "," << K  << "," << batch  << "," << is_embedded << "," << is_forward << "," << max_time << "," << diff << endl;
  //   }
  // }

  // TODO: copy more for C2R?
  if (is_forward) { // R2C
    int Mdim = (Mo/2+1)/p;
    if ((Mo/2 + 1) % p) {
      Mdim += 1;
    }

    DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);

    double *first_elems = (double *) malloc(sizeof(double) * batch);
    for (int b = 0; b < batch; b++) {
      first_elems[b] = {};
    }

    for (int l = 0; l < Ki/p; l++) {
      for (int j = 0; j < Ni; j++) {
        for (int i = 0; i < Mo; i++) {
          for (int b = 0; b < batch; b++) {
            first_elems[b] += host_in[(l * Ni*Mo + j * Mo + i)*batch + b];
          }
        }
      }
    }

    for (int b = 0; b < batch; b++) {
      MPI_Allreduce(first_elems + b, first_elems + b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    if (rank == 0) {
      printf("diff:\n");
      for (int b = 0; b < batch; b++) {
        printf("%16f %16f %16f\n", first_elems[b], host_out[b].real(), first_elems[b] - host_out[b].real());
      }
      printf("\n");
    }
    free(first_elems);
  } else { // C2R
    // TODO: fix this
    int Mdim = (Mo/2+1)/p;
    if ((Mo/2 + 1) % p) {
      Mdim += 1;
    }

    DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);
    if (rank == 0) {
      printf("diff:\n");
      for (int b = 0; b < batch; b++) {
        printf("%12f\t", (double) (M * N * K * (b+1)) - host_out[b].real());
      }
      printf("\n");
    }
  }

  fftx_plan_destroy(plan);

  DEVICE_FREE(dev_in);
  DEVICE_FREE(dev_out);

  free(host_in);
  free(host_out);

  MPI_Finalize();

  return 0;
}
