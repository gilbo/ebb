#ifndef _LISZT_MESH_H
#define _LISZT_MESH_H

extern "C" {
#include "mesh_crs.h"
}
#include "MeshIO/FacetEdgeBuilder.h"

namespace MeshIO {
	struct FacetEdgeBuilder;
}

void lMeshInitFromFacetEdgeBuilder(Mesh * mesh, MeshIO::FacetEdgeBuilder * b);
void lMeshFreeData(Mesh * mesh); //TODO: NYI

#endif
