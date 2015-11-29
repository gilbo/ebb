#include "11___c_header.h"
#include "ebb/lib/dld.h"
// In order to get access to the struct layout, we include the C-header file
// description of DLDs from the Ebb standard library.

// Notice that this version of the `tile_shuffle()` function also reads the
// dimensions of the 2d grid, rather than assuming it will be 40x40.
void tile_shuffle( void * dld_ptr ) {
  DLD *dld        = (DLD*)(dld_ptr);
  double *t_ptr   = (double*)(dld->address);
  int xstride     = (int)(dld->dim_stride[0]);
  int ystride     = (int)(dld->dim_stride[1]);
  int xdim        = (int)(dld->dim_size[0]);
  int ydim        = (int)(dld->dim_size[0]);
  int halfx       = xdim / 2;
  int halfy       = ydim / 2;

  for (int y=0; y<halfy; y++) {
    for (int x=0; x<halfx; x++) {
      int t1_idx  =         x * xstride   +         y * ystride;
      int t2_idx  = (x+halfx) * xstride   + (y+halfy) * ystride;

      double temp   = t_ptr[t1_idx];
      t_ptr[t1_idx] = t_ptr[t2_idx];
      t_ptr[t2_idx] = temp;
    }
  }
}

