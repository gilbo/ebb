
# Publication Code

This sub-directory contains code used in the 2015 ToG submission of Ebb.

# Disclaimer

The code is not guaranteed to run without some effort.  Furthermore, the numbers may be different (both better and worse) than in publication due to ongoing development of Ebb.  If you are trying to reproduce the paper results exactly, please contact the developers for more assistance.


# Applications

## FluidsGL

You will need to have CUDA and CUDA FFT installed to run this example.  You may have to fuss with paths a bit to get CUDA FFT to link correctly.  The `UseGPU` flag at the top is NOT the same thing as setting the `--gpu` command line flag.  The GPU will be used either way.  In one mode the data is copied from CPU to GPU to perform the FFT and in the other mode, the data is all assumed to be GPU resident already.

## LULESH

Lulesh code can be found in its [own example directory](../lulesh).

## VEGA

This code can be found inside [INTERNAL_devapps/fem](../../INTERNAL_devapps/fem).