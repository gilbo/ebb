#ifndef __DLD_H_
#define __DLD_H_

#include "stdint.h"

typedef struct DLD {
  uint8_t         version[2];             /* This is version 1,0 */
  uint16_t        base_type;              /* enumeration / flags */
  uint8_t         location;               /* enumeration */
  uint8_t         type_stride;            /* in bytes */
  uint8_t         type_dims[2];           /* 1,1 scalar; n,1 vector; ... */

  uint64_t        address;                /* void* */
  uint64_t        dim_size[3];            /* size in each dimension */
  uint64_t        dim_stride[3];          /* 1 = 1 element, not 1 byte */
} DLD;

/*
  The version bits are intended to provide a way for future updates to
  this specification to clearly signal that they are using a different
  specification.  In particular, if the first byte is different than
  the recipient expects, or if the later byte is greater than the
  recipient expects, then the recipient should not expect to be able to
  correctly read or write the data.
*/

/*
  For clarity, let's describe the stride parameters and how they describe
  the layout.
  
  type_stride allows us to describe alignment of each element of the
  overall array being describing.  This can be important, for example when
  the element type is a vector of 3 floats (i.e. base_type is float and
  type_dims is 3,1) but these triples are aligned to 128-bit boundaries
  (type_stride would be 16) vs the triples being densely packed
  (type_stride would be 12).

  type_dims is n,1 for vector data and n,m for matrix data.  Matrices are
  always stored in row-major layout.

  The array of data may be linear or grid-structured.  If it's linear, then
  dim_size = N,1,1 and dim_stride = 1,_,_ .  If the data is grid-structured,
  then either it's 2d: N,M,1 or 3d: N,M,L .  In either case, the dim_stride
  values control the row vs. column major layout decision as well as possible
  padding.  This allows for describing sub-rectangles of a grid as well.
*/

/*
  Two of the DLD fields describe enumerations, whose values are described
  below.

  The Location enumeration is very simple right now. There are two values:

  CPU = 0;
  GPU = 1;
*/

#define DLD_CPU 0
#define DLD_GPU 1

/*
  The Base Type enumeration is considerably more complicated.
  We want to be able to encode signed and unsigned integers, floating-point
  values, tightly-packed key data and bits.  We use a system of flags
  that can be encoded in a 16-bit value to represent these options.

  |_ _ _ _|_|_|_|_|   |_ _|_ _|_ _|_ _|
      ^    ^ ^ ^ ^      ^   ^   ^   ^
      |    | | | |      |   |   |   |
      |    | | | |      |   |   |    \__ last   dimension # bits (8,16,32,64)
      |    | | | |      |   |   \_______ middle dimension # bits
      |    | | | |      |    \__________ first  dimension # bits
      |    | | | |       \______________ # key dimensions (0 == not key)
      |    | | | |
      |    | | |  \__ if set, this is a signed integer
      |    | |  \____ if set, this is data is tightly packed bits
      |    |  \______ if set, this is a 32-bit single-precision float
      |     \________ if set, this is a 64-bit double-precision float
       \_____________ unused bits

  Here are the basic type enumerations and some example keys

  UINT_8        = 0x0
  UINT_16       = 0x1
  UINT_32       = 0x2
  UINT_64       = 0x3

  SINT_8        = 0x100
  SINT_16       = 0x101
  SINT_32       = 0x102
  SINT_64       = 0x103

  BIT           = 0x200
  FLOAT         = 0x400
  DOUBLE        = 0x800

  KEY_32        = 0x42   ( 01 00 00 10 )
  KEY_16_32_64  = 0xDB   ( 11 01 10 11 )
  KEY_64_32     = 0x8E   ( 10 00 11 10 )

  Because it's possible to construct types that have non-power-of-2 sizes,
  we include a type_stride parameter that describes how these values
  are aligned within a vector or matrix structured element.

  SPECIAL CASE:  BIT
  If the BIT type is used, then the type_dim[] and type_stride can be
  ignored.  Whichever dimension the bits are tightly packed along should
  have dim_stride[dim] set to ** 0 ** to indicate tight packing.

  Ebb's implementation does not currently make use of BIT, but we anticipate
  doing so for some sorts of data in the future.
*/

#define DLD_UINT_8          0x0
#define DLD_UINT_16         0x1
#define DLD_UINT_32         0x2
#define DLD_UINT_64         0x3

#define DLD_SINT_8          0x100
#define DLD_SINT_16         0x101
#define DLD_SINT_32         0x102
#define DLD_SINT_64         0x103

#define DLD_BIT             0x200
#define DLD_FLOAT           0x400
#define DLD_DOUBLE          0x800

/* These definitions can be combined to make keys */
#define DLD_1D_KEY          0x40
#define DLD_2D_KEY          0x80
#define DLD_3D_KEY          0xC0

#define DLD_KEY_0_0_8       0x00
#define DLD_KEY_0_0_16      0x01
#define DLD_KEY_0_0_32      0x02
#define DLD_KEY_0_0_64      0x03

#define DLD_KEY_0_8_0       0x00
#define DLD_KEY_0_16_0      0x04
#define DLD_KEY_0_32_0      0x08
#define DLD_KEY_0_64_0      0x0C

#define DLD_KEY_8_0_0       0x00
#define DLD_KEY_16_0_0      0x10
#define DLD_KEY_32_0_0      0x20
#define DLD_KEY_64_0_0      0x30

/* Here are the example keys from the long comment */
#define DLD_KEY_32          ( DLD_KEY_1D | DLD_KEY_0_0_32 )
#define DLD_KEY_64_32       ( DLD_KEY_2D | DLD_KEY_0_64_0 | DLD_KEY_0_0_32 )
#define DLD_KEY_16_32_64    ( DLD_KEY_3D | DLD_KEY_16_0_0 | \
                                           DLD_KEY_0_32_0 | DLD_KEY_0_0_64 )


#endif
/* matches ifndef __DLD_H_ */
