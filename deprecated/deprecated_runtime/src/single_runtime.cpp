
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>

extern "C" {
#include "single_runtime.h"
}

#include "boundary_set_reader.h"
#include "mesh.h"
#include "print_context.h"
#include "runtime_util.h"
#include "nested_topology.h"
#include "MeshIO/LisztFileReader.h"
#include "MeshIO/LisztFileWriter.h"

#define L_KERNEL L_ALWAYS_INLINE
#define L_NESTED L_ALWAYS_INLINE
#define L_UNNESTED
#define L_STENCIL

//define opaque data types and implementation for specific runtime

struct lContext {
	lFields * fields;
	lSets * sets;
	lScalars * scalars;
	MeshIO::LisztFileReader mesh_reader; //we hang on to this so we can load the positions on demand
	MeshIO::LisztFileWriter mesh_writer; 
	Mesh mesh;
	BoundarySetReader boundary_reader;
	lkElement active_element;
	PrintContext print_context;
};

struct lSet {
	BoundarySetReader::range_list_t ranges;
	size_t size;
};

lSet *lNewlSet () {
	lSet *set = new lSet;
	return set;
}

void lFreelSet (lSet *set) { 
	delete set; 
}

struct lkSet {
	lNestedSet set;
};

struct lkIterator {
	lNestedIterator it;
};


L_ALWAYS_INLINE 
static inline lField * unnested(lkField * field) {
	return reinterpret_cast<lField*>(field);
}
L_ALWAYS_INLINE 
static inline lSet * unnested(lkSet * set) {
	return reinterpret_cast<lSet*>(set);
}
L_ALWAYS_INLINE 
static inline lContext * unnested(lkContext * ctx) {
	return ctx->ctx;
}
L_ALWAYS_INLINE 
static inline lScalar * unnested(lkScalar * scalar) {
	return reinterpret_cast<lScalar*>(scalar);
}

L_ALWAYS_INLINE 
size_t numberOfElementsOfType(lContext * ctx, lElementType typ) {
	switch(typ) {
		case L_VERTEX: return ctx->mesh.nvertices;
		case L_EDGE:   return ctx->mesh.nedges;
		case L_FACE:   return ctx->mesh.nfaces;
		case L_CELL:   return ctx->mesh.ncells;
	}
	return 0; //silence warnings
}

//Unnested Runtime Calls

struct Mesh* lMeshFromContext(lContext *ctx) {
	return &(ctx->mesh);
}

lContext *lLoadContext (char *mesh_file) {
	lContext *ctx = new lContext;
	FILE *file = NULL;
	ctx->mesh_reader.init(mesh_file, &file);
	ctx->mesh_writer.init(mesh_file, &file);

	MeshIO::FileFacetEdge *ffe   = ctx->mesh_reader.facetEdges();
	const MeshIO::LisztHeader &h = ctx->mesh_reader.header();
	
	MeshIO::FacetEdgeBuilder builder;
	builder.init(h.nV,h.nE,h.nF,h.nC,h.nFE);
	builder.insert(0,h.nFE,ffe);
	//builder.validateFullMesh();
	ctx->mesh_reader.free(ffe);
	lMeshInitFromFacetEdgeBuilder(&ctx->mesh,&builder);
	ctx->boundary_reader.init(h.nBoundaries,ctx->mesh_reader.boundaries());

	return ctx;
}

void * lLoadPosition (lContext *ctx)  {
    assert(!"NOT IMPLEMENTED - lLoadPosition");
    return NULL;
}

lField *lLoadField(lContext *ctx, const char *key, lElementType key_type, lType val_type, size_t val_length) {
    assert(!"NOT IMPLEMENTED - lLoadField");
	return NULL;
}

lField *lInitField (lContext *ctx, lElementType key_type, lType val_type, size_t val_length) {
	lField *field = (lField*) malloc(sizeof(lField));
	size_t n_elems = numberOfElementsOfType(ctx,key_type);
	field->data = (byte*) malloc(n_elems * val_length * lUtilTypeSize(val_type));
	field->lkfield = (lkField *) field;
	return field;
}

lScalar *lInitScalar (lContext *ctx, enum lType val_type, size_t val_length) {
	lScalar *scalar = (lScalar*) malloc(sizeof(lField));
	scalar->data = (byte *) malloc(lUtilTypeSize(val_type) * val_length);
	scalar->lkscalar = (lkScalar *) scalar;
	return scalar;
}

void lExec(void (*entry_point)(lContext*),void (*entry_stencil)(lsFunctionTable*,lsContext*), lProgramArguments * arguments, size_t n_fields, size_t n_sets, size_t n_scalars) {
	lContext ctx;
	
	//allocate memory to hold global state
	ctx.fields  = (lFields*)  malloc(sizeof(lField)  * n_fields);
	ctx.sets    = (lSets*)    malloc(sizeof(lSet)    * n_sets);
	ctx.scalars = (lScalars*) malloc(sizeof(lScalar) * n_scalars);
	
	//load mesh and boundary sets
	
  FILE *file = NULL;
	ctx.mesh_reader.init(arguments->mesh_file, &file);
	ctx.mesh_writer.init(arguments->mesh_file, &file);

	MeshIO::FileFacetEdge * ffe = ctx.mesh_reader.facetEdges();
	const MeshIO::LisztHeader & h = ctx.mesh_reader.header();
	
	MeshIO::FacetEdgeBuilder builder;
	builder.init(h.nV,h.nE,h.nF,h.nC,h.nFE);
	builder.insert(0,h.nFE,ffe);
	//builder.validateFullMesh();
	ctx.mesh_reader.free(ffe);
	lMeshInitFromFacetEdgeBuilder(&ctx.mesh,&builder);
	ctx.boundary_reader.init(h.nBoundaries,ctx.mesh_reader.boundaries());
	
	//launch the program
	
	entry_point(&ctx);
	
	//do cleanup
}
L_ALWAYS_INLINE
lFields * lGetFields(lContext * ctx) {
	return ctx->fields;
}
L_ALWAYS_INLINE
lSets * lGetSets(lContext * ctx) {
	return ctx->sets;
}
L_ALWAYS_INLINE
lScalars * lGetScalars(lContext * ctx) {
	return ctx->scalars;
}

L_ALWAYS_INLINE
void lFieldBroadcast(lContext * ctx, lField * field, lElementType key_type, lType val_type, size_t val_length, void * data) {
	size_t n_elems = numberOfElementsOfType(ctx,key_type);
	for(size_t i = 0; i < n_elems; i++) {
		lkElement e = { i };
		lkFieldWrite(reinterpret_cast<lkField*>(field),e,L_ASSIGN, val_type, val_length, 0, val_length, data);
	}
}

L_RUNTIME_UNNESTED void lFieldSaveData(lContext * ctx, lField * field, lElementType key_type, lType val_type, size_t val_length, const char * key) {
    assert(!"NOT IMPLEMENTED - lFieldSaveData");
}

void lFieldEnterPhase(lField * field, lType val_type, size_t val_length, lPhase phase) {
	//nop
}

void lScalarInit(lContext * ctx, lScalar * scalar, lType val_type, size_t val_length) {
	scalar->data = (byte*) malloc(lUtilTypeSize(val_type) * val_length);
}


L_ALWAYS_INLINE 
void lScalarRead(lContext * ctx, lScalar * scalar, lType element_type, size_t element_length, size_t offset, size_t val_length, void * result) {
	size_t scalar_size = lUtilTypeSize(element_type);
	memcpy(result,scalar->data + offset * scalar_size, scalar_size * val_length);
}

L_ALWAYS_INLINE 
void lScalarWrite(lContext * ctx, lScalar * scalar, lReduction reduction, lType element_type, size_t element_length, size_t offset, size_t val_length, void * value) {
	size_t scalar_size = lUtilTypeSize(element_type);
	lUtilValueReduce(scalar->data + offset * scalar_size, reduction, element_type, val_length, value);
}

void lScalarEnterPhase(lScalar * scalar, lType val_type, size_t val_length, lPhase phase) {
	//nop
}

void lSetInitBoundary(lContext * ctx, lSet * set, lElementType type, const char * boundary_name) {
    assert(!"NOT IMPLEMENTED - lSetInitBoundary");
}


void *lLoadBoundarySet (lContext *ctx, lElementType type, const char *boundary_name, size_t *bsize) {
	lSet set;
	lSetInitBoundary(ctx, &set, type, boundary_name);
	*bsize = set.size;

	RTYPE *data = (RTYPE *) malloc(set.size * sizeof(RTYPE));
	RTYPE *d    = data;
	for (int i = 0; i < set.ranges.size(); i++)
		for (int j = set.ranges[i].first; j < set.ranges[i].second; j++)
			d[0] = j, d++;

	return data;
}

L_ALWAYS_INLINE 
size_t lSetSize(lContext * ctx, lSet * set) {
	return set->size;
}

L_ALWAYS_INLINE
void lKernelRun(lContext * ctx,  lSet * set, lElementType typ, int id, void (*kernel)(lkContext)) {

#ifndef NDEBUG
	ctx->print_context.push();
#endif

	for(size_t i = 0; i < set->ranges.size(); i++) {
		uint32_t start = set->ranges[i].first;
		uint32_t end = set->ranges[i].second;
		for(size_t j = start; j < end; j++) {

#ifndef NDEBUG
			ctx->print_context.mark(typ, j);
#endif
			lkContext kctx;
			kctx.ctx  = ctx;
			kctx.element.data = j;
			kernel(kctx);
		}
	}
	
#ifndef NDEBUG
	ctx->print_context.pop();
#endif

}
/*kernels have the signature and structure: 
void kernel_name(lkContext * ctx) {
	lkElement e;
	if(lkGetActiveElement(ctx,&e)) {
	  <...>
	}
}*/

void lPrintBegin(lContext * ctx) {
#ifndef NDEBUG
	std::cout << ctx->print_context;
#endif
	std::cout << "liszt: ";
}
void lPrintEnd(lContext * ctx) {
	std::cout << "\n";
}

void printValue(lType typ, const void * value) {
	switch(typ) {
		case L_FLOAT:
			std::cout << *(float*)value;
			break;
		case L_INT:
			std::cout << *(int*)value;
			break;
		case L_BOOL:
			std::cout << *(bool*)value;
			break;
		case L_DOUBLE:
			std::cout << *(double*)value;
			break;
		case L_STRING:
			std::cout << (const char *) value;
			break;
	}
}

void lPrintValue(lContext * ctx, lType typ, size_t r, size_t c, const void * value) {
	byte * v = (byte *) value;
	size_t size = lUtilTypeSize(typ);
	if(c != 0) { 
		std::cout << "[";
		for(size_t i = 0; i < r; i++) {
			for (size_t j = 0; j < c; j++) {
				printValue(typ,v);
				v += size;
				if(j < c - 1) std::cout << ",";
			}
			if(i < r - 1) std::cout << "|";
			else std::cout << "]";  
		}
	} else if(r != 0) {
		std::cout << "(";
		for(int i = 0; i < r - 1; i++) {
			printValue(typ,v);
			std::cout << ",";
			v += size;
		}
		printValue(typ,v);
		std::cout << ")";
	} else {
		printValue(typ,v);
	}
}
double lWallTime(lContext * ctx) { //counter, time reported in seconds
	timeval currentTime;
	gettimeofday(&currentTime, 0);
	return currentTime.tv_sec + 1e-6 * currentTime.tv_usec;
}
//Nested Runtime Calls
void lkPrintBegin(lkContext * ctx) {
	lPrintBegin(unnested(ctx));
}
void lkPrintEnd(lkContext * ctx) {
	lPrintEnd(unnested(ctx));
}
void lkPrintValue(lkContext * ctx, lType typ, size_t r, size_t c, const void * value) {
	lPrintValue(unnested(ctx), typ, r, c, value);
}
void lkPrintElement(lkContext * ctx, lElementType typ, lkElement e) {
	switch(typ) {
		case L_VERTEX:
			printf("Vertex ");
			break;
		case L_EDGE:
			printf("Edge ");
			break;
		case L_FACE:
			printf("Face ");
			break;
		case L_CELL:
			printf("Cell ");
			break;
	}
	std::cout << elementID(e);
}

L_ALWAYS_INLINE 
lkFields * lkGetFields(lkContext * ctx) {
	return reinterpret_cast<lkFields*>(unnested(ctx)->fields);
}

L_ALWAYS_INLINE 
lkScalars * lkGetScalars(lkContext * ctx) {
	return reinterpret_cast<lkScalars*>(unnested(ctx)->scalars);
}

L_ALWAYS_INLINE 
BOOL lkGetActiveElement(lkContext * ctx, lkElement * e) {
	*e = ctx->element;
	return true;
}

void lkFieldRead(lkField * scalar, lkElement e, lType element_type, size_t element_length, size_t val_offset, size_t val_length,  void * result) {
	lField * f = unnested(scalar);
	size_t offset,size;
	lUtilValueLocation(elementID(e),element_type,element_length,val_offset,val_length,&offset,&size);
	memcpy(result,f->data + offset, size);
}

L_ALWAYS_INLINE 
void lkFieldWrite(lkField * scalar, lkElement e, lReduction reduction, lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * value) {
	lField * f = unnested(scalar);
	size_t offset,size;
	lUtilValueLocation(elementID(e),element_type,element_length,val_offset,val_length,&offset,&size);
	lUtilValueReduce(f->data + offset, reduction, element_type, val_length, value);
}

L_ALWAYS_INLINE
void lkScalarRead(lkContext * ctx, lkScalar * scalar, lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * result){
	lScalarRead(unnested(ctx),unnested(scalar),element_type, element_length, val_offset, val_length, result);
}

L_ALWAYS_INLINE
void lkScalarWrite(lkContext * ctx, lkScalar * scalar, lReduction reduction, lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * value) {
	lScalarWrite(unnested(ctx),unnested(scalar),reduction,element_type, element_length, val_offset, val_length, value);
}

L_ALWAYS_INLINE 
size_t lkSetSize(lkContext * ctx, lkSet * set) {
	return set->set.size;
}
L_ALWAYS_INLINE
void lkSetGetIterator(lkContext * ctx, lkSet * set, lElementType typ, lkIterator * iterator) {
	lNestedSetGetIterator(&set->set,&iterator->it);
#ifndef NDEBUG
	unnested(ctx)->print_context.push();
#endif
}

L_ALWAYS_INLINE
BOOL lkIteratorNext(lkContext * ctx, lkIterator * iterator, lElementType typ, lkElement * e, int * lbl) {
	bool result = lNestedIteratorNext(&iterator->it,e,lbl);
#ifndef NDEBUG	
	if(result) {
		unnested(ctx)->print_context.mark(typ, elementID(*e));
	} else {
		unnested(ctx)->print_context.pop();
	}
#endif
	return result;
}

//now all the topological functions ....
void lVerticesOfMesh(lContext * ctx, lSet * set) {
	set->size = ctx->mesh.nvertices;
	set->ranges.push_back(std::make_pair(0, set->size));
}
void lEdgesOfMesh(lContext * ctx, lSet * set) {
	set->size = ctx->mesh.nedges;
	set->ranges.push_back(std::make_pair(0, set->size));
}
void lFacesOfMesh(lContext * ctx, lSet * set) {
	set->size = ctx->mesh.nfaces;
	set->ranges.push_back(std::make_pair(0, set->size));
}
void lCellsOfMesh(lContext * ctx, lSet * set) {
	set->size = ctx->mesh.ncells - 1;
	set->ranges.push_back(std::make_pair(1, ctx->mesh.ncells));
}

#define NESTED_TOPOLOGY_GET_MESH(ctx) (&unnested(ctx)->mesh)
#define NESTED_TOPOLOGY_GET_SET(set) (set->set)
#include "nested_topology_functions.inc"


int lkIDOfVertex(lkContext * ctx, lkElement e) {
	return elementID(e);
}
int lkIDOfEdge(lkContext * ctx, lkElement e) {
	return elementID(e);
}
int lkIDOfFace(lkContext * ctx, lkElement e) {
	return elementID(e);
}
int lkIDOfCell(lkContext * ctx, lkElement e) {
	return elementID(e);
}

#include "vector.h"
#include "matrix.h"
#include "liszt_math.h"


//code generated from the liszt compiler is now included here after runtime-specific
//headers have been included

//#include "generated.cpp"

