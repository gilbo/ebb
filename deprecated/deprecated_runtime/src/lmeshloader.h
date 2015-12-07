#ifndef _LMESH_LOADER_H
#define _LMESH_LOADER_H

#include "mesh_crs.h"


struct LMeshBoundary {
    char * name;
    const char * type; //e.g. "vertex"
    uint64_t * data;
    size_t size;
};
struct LMeshField {
    char * name;
    void * data;
    const char * elemtype; //e.g. "vertices"
    const char * datatype; //e.g. "int", "bool", "float", or "double"
    size_t nelems; //1 == scalar, N == vector of length N
};
struct LMeshData {
    struct Mesh mesh;
    int nBoundaries;
    int nFields;
    struct LMeshBoundary * boundaries;
    struct LMeshField * fields;
    struct LMeshLoader * loader;
};
void LMeshLoadFromFile(const char * filename, struct LMeshData * meshdata);
void LMeshFree(struct LMeshData * data);

#endif