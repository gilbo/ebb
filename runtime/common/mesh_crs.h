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

	struct CRS vtov;
	struct CRS vtoe;
	struct CRS vtof;
	struct CRS vtoc;

	struct CRSConst etov;
	struct CRS etof;
	struct CRS etoc;

	struct CRS ftov;
	struct CRS ftoe;
	struct CRSConst ftoc;

	struct CRS ctov;
	struct CRS ctoe;
	struct CRS ctof;
	struct CRS ctoc;
};

struct Mesh* lMeshInitFromFile(const char *mesh_file);

const unsigned int FLIP_DIRECTION_SHIFT = 31;
const unsigned int FLIP_DIRECTION_BIT = 1u << FLIP_DIRECTION_SHIFT;

#endif
