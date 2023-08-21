#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
#include "fftx_util.h"
#include "fftx_mpi.hpp"

#include "interface.hpp"
#include "batch1ddftObj.hpp"
#include "ibatch1ddftObj.hpp"
#if defined FFTX_CUDA
#include "cudabackend.hpp"
#elif defined FFTX_HIP
#include "hipbackend.hpp"
#else
#include "cpubackend.hpp"
#endif


using namespace std;

fftx_plan fftx_plan_distributed_spiral(int r, int c, int M, int N, int K, int batch, bool is_embedded, bool is_complex) {

  fftx_plan plan = (fftx_plan) malloc(sizeof(fftx_plan_t));

  plan->b = batch;
  plan->is_embed = is_embedded;
  plan->is_complex = is_complex;
  plan->M = M;
  plan->N = N;
  plan->K = K;

  init_2d_comms(plan, r, c,  M,  N, K);   //embedding uses the input sizes

  DEVICE_MALLOC(&(plan->Q3), M*N*K*(is_embedded ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);
  DEVICE_MALLOC(&(plan->Q4), M*N*K*(is_embedded ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);

  int batch_sizeZ = M/r * N/c;
  int batch_sizeX = N/c * K/r;
  int batch_sizeY = K/r * M/c;

  int inK = K * (is_embedded ? 2 : 1);
  int inM = M * (is_embedded ? 2 : 1);
  int inN = N * (is_embedded ? 2 : 1);


  // int outK = K * (is_embedded ? 2 : 1);
  // int outM = M * (is_embedded ? 2 : 1);
  // int outN = N * (is_embedded ? 2 : 1);


  batch_sizeX *= (is_embedded ? 2 : 1);
  batch_sizeY *= (is_embedded ? 4 : 1);


  // if ((plan->is_complex))
  //   {
  //     //read seq write strided
  //     DEVICE_FFT_PLAN_MANY(&(plan->stg1), 1, &inK,
	// 		   &inK,             plan->b, inK*plan->b,
	// 		   &inK, batch_sizeZ*plan->b, plan->b,
	// 		   DEVICE_FFT_Z2Z, batch_sizeZ);

  //     //inverse plan -> read strided write seq
  //     DEVICE_FFT_PLAN_MANY(&(plan->stg1i), 1, &inK,
	// 		   &inK, batch_sizeZ*plan->b, plan->b,
	// 		   &inK,             plan->b, inK*plan->b,
	// 		   DEVICE_FFT_Z2Z, batch_sizeZ);

  //   }
  // else
  //   {
  //     //read seq write strided
  //     DEVICE_FFT_PLAN_MANY(&(plan->stg1), 1, &inK,
	// 		   &inK,             plan->b, inK*plan->b,
	// 		   &inK, batch_sizeZ*plan->b, plan->b,
	// 		   DEVICE_FFT_D2Z, batch_sizeZ);

  //     //inverse plan -> read strided write seq
  //     DEVICE_FFT_PLAN_MANY(&(plan->stg1i), 1, &inK,
	// 		   &inK, batch_sizeZ*plan->b, plan->b,
	// 		   &inK,             plan->b, inK*plan->b,
	// 		   DEVICE_FFT_Z2D, batch_sizeZ);
  //   }

  // //read seq write strided
  // DEVICE_FFT_PLAN_MANY(&(plan->stg2), 1, &inM,
	// 	       &inM,           plan->b, inM*plan->b,
	// 	       &inM, batch_sizeX*plan->b, plan->b,
	// 	       DEVICE_FFT_Z2Z, batch_sizeX);

  // //read seq write seq
  // DEVICE_FFT_PLAN_MANY(&(plan->stg3), 1, &inN,
	// 	       &inN, plan->b, inN*plan->b,
	// 	       &inN, plan->b, inN*plan->b,
	// 	       DEVICE_FFT_Z2Z, batch_sizeY);

  // //read strided write seq
  // DEVICE_FFT_PLAN_MANY(&(plan->stg2i), 1, &inM,
	// 	       &inM, batch_sizeX*plan->b, plan->b,
	// 	       &inM,           plan->b, inM*plan->b,
	// 	       DEVICE_FFT_Z2Z, batch_sizeX);

  return plan;
}

void fftx_execute_spiral(fftx_plan plan, double* out_buffer, double*in_buffer, int direction)
{
  int batch_sizeZ = plan->M/plan->r * plan->N/plan->c;
  int batch_sizeX = plan->N/plan->c * plan->K/plan->r;
  int batch_sizeY = plan->K/plan->r * plan->M/plan->c;

  int inK = plan->K * (plan->is_embed ? 2 : 1);
  int inM = plan->M * (plan->is_embed ? 2 : 1);
  int inN = plan->N * (plan->is_embed ? 2 : 1);

  batch_sizeX *= (plan->is_embed ? 2 : 1);
  batch_sizeY *= (plan->is_embed ? 4 : 1);

  std::vector<int> size_stg1 = {inK, batch_sizeZ, 0, 1};  
  BATCH1DDFTProblem bdstg1(size_stg1, "b1dft");
  std::vector<int> size_stg2 = {inM, batch_sizeX, 0, 1};  
  BATCH1DDFTProblem bdstg2(size_stg2, "b1dft");
  std::vector<int> size_stg3 = {inN, batch_sizeY, 0, 0};  
  BATCH1DDFTProblem bdstg3(size_stg3, "b1dft");	

  std::vector<int> size_istg1 = {inK, batch_sizeZ, 1, 0};
  IBATCH1DDFTProblem ibdstg1(size_istg1, "ib1dft");
  std::vector<int> size_istg2 = {inM, batch_sizeX, 1, 0};  
  IBATCH1DDFTProblem ibdstg2(size_istg2, "ib1dft");
  

  if (direction == DEVICE_FFT_FORWARD) {
    if (plan->is_complex) {
      for (int i = 0; i != plan->b; ++i) {
        // if(use_fftx) {
          std::vector<void*> args{plan->Q3 + i, in_buffer+i};
          bdstg1.setArgs(args);
          bdstg1.transform();
        // } 
        // else
        //   DEVICE_FFT_EXECZ2Z(plan->stg1, ((DEVICE_FFT_DOUBLECOMPLEX  *) in_buffer + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i), direction);
      }
    }
    // } else {
    //   for (int i = 0; i != plan->b; ++i) {
    //     DEVICE_FFT_EXECD2Z(plan->stg1, ((DEVICE_FFT_DOUBLEREAL  *) in_buffer + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i));
    //   }
    // }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);

    for (int i = 0; i != plan->b; ++i) {
      // if(use_fftx) {
        std::vector<void*> args{plan->Q3 + i, plan->Q4 + i};
        bdstg2.setArgs(args);
        bdstg2.transform();
      // } 
      // else
      //   DEVICE_FFT_EXECZ2Z(plan->stg2, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i), direction);
    }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_2, plan->is_embed);

    for (int i = 0; i != plan->b; ++i) {
      // if(use_fftx) {
        std::vector<void*> args{out_buffer + i, plan->Q4 + i};
        bdstg3.setArgs(args);
        bdstg3.transform();
      // }
      // else
      //   DEVICE_FFT_EXECZ2Z(plan->stg3, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i), direction);
    }
  } else if (direction == DEVICE_FFT_INVERSE) {
    for (int i = 0; i != plan->b; ++i) {
      // if(use_fftx) {
        std::vector<void*> args{plan->Q3 + i, in_buffer + i};
        bdstg3.setArgs(args);
        bdstg3.transform();
      // } 
      // else {
      //   DEVICE_FFT_EXECZ2Z(
      //     plan->stg3,
      //     ((DEVICE_FFT_DOUBLECOMPLEX  *) in_buffer + i),
      //     ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i),
      //     direction
      //   );
      // }
    }
    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_3, plan->is_embed);
    for (int i = 0; i != plan->b; ++i){
      // if(use_fftx) {
        std::vector<void*> args{plan->Q3 + i, plan->Q4 + i};
        ibdstg2.setArgs(args);
        ibdstg2.transform();
      // }
      // else
      //   DEVICE_FFT_EXECZ2Z(plan->stg2i, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i), direction);
    }
    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_4, plan->is_embed);

    if (plan->is_complex) {
      for (int i = 0; i != plan->b; ++i) {
        // if(use_fftx) {
          std::vector<void*> args{out_buffer + i, plan->Q4 + i};
          ibdstg1.setArgs(args);
          ibdstg1.transform();
        // }
        // else
        //   DEVICE_FFT_EXECZ2Z(plan->stg1i, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i), direction);
      }
    }
    // } else { // untested
    //   for (int i = 0; i != plan->b; ++i) {
    //     DEVICE_FFT_EXECZ2D(plan->stg1i, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLEREAL  *) out_buffer + i));
    //   }
    // }
  }
}

void fftx_plan_destroy_spiral(fftx_plan plan) {
  if (plan) {
    if (plan->c == 0)
      destroy_1d_comms(plan);
    else
      destroy_2d_comms(plan);

    DEVICE_FREE(plan->Q3);
    DEVICE_FREE(plan->Q4);

    free(plan);
  }
}
