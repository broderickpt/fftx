#include <cmath> // Without this, abs is the wrong function!
#include <random>
// #include "rconv2.fftx.codegen.hpp"
// #include "rconv3.fftx.codegen.hpp"
// #include "rconv.h"

#include "RealConvolution.hpp"
#include "fftx3utilities.h"


// using namespace fftx;
// fftx::handle_t (a_transform)
template<int DIM>
void rconvDimension(std::vector<int> sizes,
                    fftx::box_t<DIM> a_domain,
                    fftx::box_t<DIM> a_fdomain,
                    int a_rounds,
                    int a_verbosity)
{
  std::cout << "***** test " << DIM << "D real convolution on "
            << a_domain << std::endl;

  // RealConvolution<DIM> fun(a_transform, a_domain, a_fdomain);
  RCONVProblem rp("rconv");
  RealConvolution<DIM> fun(rp, sizes, a_domain, a_fdomain);
  TestRealConvolution<DIM>(fun, a_rounds, a_verbosity);
}


int main(int argc, char* argv[])
{
  int mm = 24, nn = 32, kk = 40; // default cube dimensions
  char *prog = argv[0];
  int baz = 0;
  int verbosity = 0;
  int rounds = 2;
  while ( argc > 1 && argv[1][0] == '-' ) {
      switch ( argv[1][1] ) {
      case 'i':
          argv++, argc--;
          rounds = atoi ( argv[1] );
          break;
      case 's':
          argv++, argc--;
          mm = atoi ( argv[1] );
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          nn = atoi ( & argv[1][baz] );
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          kk = atoi ( & argv[1][baz] );
          break;
      case 'v':
          argv++, argc--;
          verbosity = atoi ( argv[1] );
          break;
      case 'h':
          printf ( "Usage: %s: [ -i rounds ] [-v verbosity: 0 for summary, 1 for categories, 2 for subtests, 3 for all iterations] [ -s MMxNNxKK ] [ -h (print help message) ]\n", argv[0] );
          exit (0);
      default:
          printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
      }
      argv++, argc--;
  }
  
  printf("Running size %dx%dx%d with verbosity %d, random %d rounds\n", mm, nn, kk, verbosity, rounds);
  // std::cout << mm << " " << nn << " " << kk << std::endl;
  std::vector<int> sizes{mm,nn,kk};


  /*
    Set up random number generator.
  */
  std::random_device rd;
  generator = std::mt19937(rd());
  unifRealDist = std::uniform_real_distribution<double>(-0.5, 0.5);

  /*
    2-dimensional tests.
  */
  //    rconv2::init();
  //    rconvDimension(rconv2::transform, domain2, fdomain2,
  //                   rounds, verbosity);
  //    rconv2::destroy();
  
  /*
    3-dimensional tests.
  */
  // rconv3::init();
  const int offx = 3;
  const int offy = 5;
  const int offz = 11;

  #if FFTX_COMPLEX_TRUNC_LAST
  const int fx = mm;
  const int fy = nn;
  const int fz = kk/2 + 1;
  #else
  const int fx = mm/2 + 1;
  const int fy = nn;
  const int fz = kk;
  #endif

  box_t<3> domain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                   point_t<3>({{offx+mm, offy+nn, offz+kk}}));
  box_t<3> fdomain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                    point_t<3>({{offx+fx, offy+fy, offz+fz}}));
  // std::cout << domain3 << std::endl;
  rconvDimension(sizes, domain3, fdomain3, rounds, verbosity);
  // rconv3::destroy();
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
