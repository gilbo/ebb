#ifndef __SINGLE_LISZT_RUNTIME_H
#define __SINGLE_LISZT_RUNTIME_H

#include "liszt_runtime.h"

typedef unsigned char byte;

struct lStencilData {
	void (*stencil_fn)(struct lsFunctionTable*,struct lsContext*);
	uint32_t is_trivial;
};

struct lField {
	byte * data;
	struct lkField *lkfield;
};

struct lScalar {
	byte * data;
	struct lkScalar *lkscalar;
};

//in single-core runtime these are simply the same object as the unnested version
//the 'unnested' function simply casts these objects to their unnested equivalent

struct lkField {
	byte * data;
};

struct lkScalar {
	byte * data;
};

struct lkContext {
	struct lContext * ctx;
	struct lkElement element;
};

//stenciling functions, stencils are not needed for single core so these are all blank
struct lsElement  {};
struct lsSet      {};
struct lsContext  {};
struct lsIterator {};

#endif