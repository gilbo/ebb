#ifndef _LISZT_MATH_H
#define _LISZT_MATH_H

#include "float.h"


using std::min;
using std::max;

#ifdef __CUDACC__
#define M_MODIFIERS __host__ __device__ static inline
#else
#define M_MODIFIERS static inline
#endif

M_MODIFIERS double MATH_PI() { return M_PI; }
M_MODIFIERS float MIN_FLOAT() { return FLT_MIN; }
M_MODIFIERS float MAX_FLOAT() { return FLT_MAX; }
M_MODIFIERS double MIN_DOUBLE() { return DBL_MIN; }
M_MODIFIERS double MAX_DOUBLE() { return DBL_MAX; }

#define MAKE_CONVERT(name,type) \
template<typename T> \
M_MODIFIERS type name(const T & t) { return (type) t; }

MAKE_CONVERT(toByte,unsigned char)
MAKE_CONVERT(toChar,char)
MAKE_CONVERT(toDouble,double)
MAKE_CONVERT(toFloat,float)
MAKE_CONVERT(toInt,int)
MAKE_CONVERT(toLong,long)
MAKE_CONVERT(toShort, short int)

#undef MAKE_CONVERT
#undef M_MODIFIERS
#endif
