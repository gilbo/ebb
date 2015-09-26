extern "C" {
#include "lmeshloader.h"
}
#include "MeshIO/LisztFileReader.h"
#include "mesh.h"
#include <algorithm>

static const char * ElemToString(MeshIO::IOElemType elemtype) {
	switch(elemtype) {
	    case MeshIO::VERTEX_T: return "vertices";
	    case MeshIO::EDGE_T: return "edges";
	    case MeshIO::FACE_T: return "faces";
	    case MeshIO::CELL_T: return "cells";
	    default: assert(!"Unexpected elem type"); return "unknown";
	}
}
static void addEntries(std::vector<uint64_t> * elems, MeshIO::BoundarySetEntry * entries, size_t nBoundaries, size_t bid) {
    assert(bid < nBoundaries);
    size_t start = entries[bid].start;
    size_t end = entries[bid].end;
    if(entries[bid].type & MeshIO::AGG_FLAG) {
        addEntries(elems,entries,nBoundaries,start);
        addEntries(elems,entries,nBoundaries,end);
    } else {
        for(size_t i = start; i < end; i++)
            elems->push_back(i);
    }
}
static void LoadBoundary(LMeshBoundary * to, MeshIO::BoundarySetEntry * entries, size_t nBoundaries, size_t bid) {
    MeshIO::BoundarySetEntry * from = &entries[bid];
    to->name = strdup(from->name.c_str());
    to->type = ElemToString(from->type);
    std::vector<uint64_t> elems;
    addEntries(&elems, entries, nBoundaries, bid);
    std::sort(elems.begin(), elems.end());
    to->size = elems.size();
    to->data = new uint64_t[to->size];
    memcpy(to->data,&elems[0],sizeof(uint64_t)*to->size);
}
static const char * MeshTypeToString(char t) {
    switch(t) {
        case MeshIO::LISZT_INT: return "int";
        case MeshIO::LISZT_FLOAT: return "float";
        case MeshIO::LISZT_DOUBLE: return "double";
        case MeshIO::LISZT_BOOL: return "bool";
        default: assert(!"unexpected mesh type"); return "unknown";
    }
}
static void LoadField(LMeshField * f, MeshIO::LisztFileReader * mesh_reader, size_t offset) {
    MeshIO::IOElemType elemtype;
    char datatype;
    mesh_reader->loadField(offset, &f->data,&f->name, &elemtype, &datatype, &f->nelems);
    f->elemtype = ElemToString(elemtype);
    f->datatype = MeshTypeToString(datatype);
}

void LMeshLoadFromFile(const char * filename, LMeshData * md) {
    FILE *file = NULL;
    MeshIO::LisztFileReader mesh_reader;
    mesh_reader.init(filename, &file);
    MeshIO::FileFacetEdge *ffe   = mesh_reader.facetEdges();
	const MeshIO::LisztHeader &h = mesh_reader.header();
	MeshIO::FacetEdgeBuilder builder;
	builder.init(h.nV,h.nE,h.nF,h.nC,h.nFE);
	builder.insert(0,h.nFE,ffe);
	mesh_reader.free(ffe);
	lMeshInitFromFacetEdgeBuilder(&md->mesh,&builder);	
	md->nBoundaries = h.nBoundaries;
	MeshIO::BoundarySetEntry * boundaries = mesh_reader.boundaries();
	md->boundaries = new LMeshBoundary[md->nBoundaries];
	for(size_t i = 0; i < md->nBoundaries; i++)
	    LoadBoundary(&md->boundaries[i],boundaries,md->nBoundaries,i);
	md->nFields = mesh_reader.numFields();
	md->fields = new LMeshField[md->nFields];
	for(size_t i = 0; i < md->nFields; i++)
	    LoadField(&md->fields[i],&mesh_reader,i);
}

void LMeshFree(LMeshData * md) {
    for(size_t i = 0; i < md->nBoundaries; i++) {
	    delete [] md->boundaries[i].data;
	    free(md->boundaries[i].name);
	}
	for(size_t i = 0; i < md->nFields; i++) {
	    free(md->fields[i].data);
	    free(md->boundaries[i].name);
	}
	delete [] md->boundaries;
	delete [] md->fields;
	//TODO: free the mesh
}