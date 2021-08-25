#! /bin/sh

##  Simply call python (gen_files.py) to build each library

python gen_files.py fftx_mddft cuda true
python gen_files.py fftx_mddft cuda false
python gen_files.py fftx_mdprdft cuda true
python gen_files.py fftx_mdprdft cuda false
python gen_files.py fftx_rconv cuda

##  to build HIP code:

##  python gen_files.py fftx_mddft hip true
##  python gen_files.py fftx_mddft hip false
##  python gen_files.py fftx_mdprdft hip true
##  python gen_files.py fftx_mdprdft hip false
##  python gen_files.py fftx_rconv hip
