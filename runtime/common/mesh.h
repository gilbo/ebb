#ifndef _LISZT_MESH_H
#define _LISZT_MESH_H

extern "C" {
#include "common/mesh_crs.h"
}
#include "common/MeshIO/FacetEdgeBuilder.h"

namespace MeshIO {
	struct FacetEdgeBuilder;
}

void lMeshInitFromFacetEdgeBuilder(Mesh * mesh, MeshIO::FacetEdgeBuilder * b);
void lMeshFreeData(Mesh * mesh); //TODO: NYI

#endif
