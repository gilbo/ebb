#ifndef __SINGLE_LISZT_RUNTIME_H
#define __SINGLE_LISZT_RUNTIME_H

#include "common/liszt_runtime.h"

typedef unsigned char byte;

struct lStencilData {
	void (*stencil_fn)(struct lsFunctionTable*,struct lsContext*);
	bool is_trivial;
};

struct lField {
	byte * data;
	lkField *lkfield;
};

struct lScalar {
	byte * data;
	lkScalar *lkscalar;
};

//in single-core runtime these are simply the same object as the unnested version
//the 'unnested' function simply casts these objects to their unnested equivalent

struct lkField {
	byte * data;
};

struct lkScalar {
	byte * data;
};

#endif