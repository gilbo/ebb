#ifndef _LISZT_MESH_CRS_H
#define _LISZT_MESH_CRS_H

#include "stdint.h"
#include "stdlib.h"

struct CRS {
	uint32_t * row_idx;
	uint32_t * values;
};

struct CRSConst {
	//pointer to array of two values
	uint32_t (*values)[2];
};

struct Mesh {
	size_t nvertices;
	size_t nedges;
	size_t nfaces;
	size_t ncells;

	CRS vtov;
	CRS vtoe;
	CRS vtof;
	CRS vtoc;

	CRSConst etov;
	CRS etof;
	CRS etoc;

	CRS ftov;
	CRS ftoe;
	CRSConst ftoc;

	CRS ctov;
	CRS ctoe;
	CRS ctof;
	CRS ctoc;
};

const unsigned int FLIP_DIRECTION_SHIFT = 31;
const unsigned int FLIP_DIRECTION_BIT = 1u << FLIP_DIRECTION_SHIFT;

#endif
