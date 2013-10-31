#ifndef _LISZT_RUNTIME_H
#define _LISZT_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
// #include <string>

typedef uint8_t BOOL;

enum lElementType {
	L_VERTEX = 0,
	L_CELL   = 1,
	L_EDGE   = 2,
	L_FACE   = 3,
};
enum {
	L_ELEMENT_TYPE_SIZE = 4
};
enum lType {
	L_INT = 0,
	L_FLOAT,
	L_DOUBLE,
	L_BOOL,
	L_STRING
};

enum lReduction {
    L_ASSIGN = 0,
	L_PLUS,
	L_MINUS,
	L_MULTIPLY,
	L_DIVIDE,
	L_MIN,
	L_MAX,
	L_BOR,
	L_BAND,
	L_XOR,
	L_AND,
	L_OR
};

enum lPhase {
	L_READ_ONLY = 0,
	L_WRITE_ONLY,
	L_MODIFY,
	L_REDUCE_PLUS,
	L_REDUCE_MULTIPLY,
	L_REDUCE_MIN,
	L_REDUCE_MAX,
	L_REDUCE_BOR,
	L_REDUCE_BAND,
	L_REDUCE_XOR,
	L_REDUCE_AND,
	L_REDUCE_OR
};
struct lContext;
struct lField;
struct lScalar;
struct lSet;

struct lFields;
/* 
{
   lField flux;
   lField temp;
   .... //compiler will will this in
};
*/
struct lScalars;
/* 
{
   lScalar error;
   .... //compiler will will this in
};
*/
struct lSets;
/*
{
   lSet boundary;
   ..... //compiler will will this in
}*/

struct lkField;
struct lkFields;

struct lkScalar;
struct lkScalars;

struct lkContext;

//for efficiency we define lkElement explicitly for all runtimes as a 32-bit value
struct lkElement { uint32_t data; };

struct lkSet;
struct lkIterator;

struct lsContext;
struct lsElement;
struct lsSet;
struct lsIterator;

struct lsFunctionTable;

struct lStencilData;
struct Mesh;

//Modifiers for runtime functions, GPU modifiers are set rather than in GPU specific code to simplify definition order for C code
#ifdef __CUDACC__
#define L_RUNTIME_NESTED __device__
#define L_RUNTIME_UNNESTED __host__
#define L_RUNTIME_STENCIL
#define L_RUNTIME_ALL __device__ __host__
#else
#define L_RUNTIME_NESTED
#define L_RUNTIME_UNNESTED
#define L_RUNTIME_STENCIL
#define L_RUNTIME_ALL
#endif

//Unnested Runtime Calls

struct lProgramArguments;

// void lExec(void (*entry_point)(struct lContext*),void (*entry_stencil)(struct lsFunctionTable*, struct lsContext*), struct lProgramArguments * arguments, size_t n_fields, size_t n_sets, size_t n_scalars);

// L_RUNTIME_UNNESTED struct lFields  * lGetFields(struct lContext * ctx);
// L_RUNTIME_UNNESTED struct lSets    * lGetSets(struct lContext * ctx);
// L_RUNTIME_UNNESTED struct lScalars * lGetScalars(struct lContext * ctx);

// move this into lua
// L_RUNTIME_UNNESTED void lFieldInit(struct lContext * ctx, struct lField * field, int id, enum lElementType key_type, enum lType val_type, size_t val_length);

// change signatures and generate code for each type

// Load a context, which stores a bunch of necessary per-mesh instantiated objects
L_RUNTIME_UNNESTED struct lContext *lLoadContext (char *mesh_file);
L_RUNTIME_UNNESTED struct Mesh     *lMeshFromContext (struct lContext *ctx);
L_RUNTIME_UNNESTED void *lLoadPosition(struct lContext *ctx);
L_RUNTIME_UNNESTED struct lField   *lLoadField   (struct lContext *ctx, const char *key, enum lElementType key_type, enum lType val_type, size_t val_length);
L_RUNTIME_UNNESTED struct lField   *lInitField   (struct lContext *ctx, enum lElementType key_type, enum lType val_type, size_t val_length);
L_RUNTIME_UNNESTED struct lScalar  *lInitScalar  (struct lContext *ctx, enum lType val_type, size_t val_length);

L_RUNTIME_UNNESTED struct lSet *lNewlSet ();
L_RUNTIME_UNNESTED void lFreelSet (struct lSet *set);

L_RUNTIME_UNNESTED uint32_t lNumVertices (struct lContext *ctx);
L_RUNTIME_UNNESTED uint32_t lNumEdges    (struct lContext *ctx);
L_RUNTIME_UNNESTED uint32_t lNumFaces    (struct lContext *ctx);
L_RUNTIME_UNNESTED uint32_t lNumCells    (struct lContext *ctx);

L_RUNTIME_UNNESTED void lFieldBroadcast (struct lContext * ctx, struct lField * field, enum lElementType key_type, enum lType val_type, size_t val_length, void * data);
L_RUNTIME_UNNESTED void lFieldLoadData  (struct lContext * ctx, struct lField * field, enum lElementType key_type, enum lType val_type, size_t val_length, const char * key);
L_RUNTIME_UNNESTED void lFieldSaveData  (struct lContext * ctx, struct lField * field, enum lElementType key_type, enum lType val_type, size_t val_length, const char * key);


L_RUNTIME_UNNESTED void lFieldEnterPhase (struct lField * field, enum lType val_type, size_t val_length, enum lPhase phase);

// L_RUNTIME_UNNESTED void lScalarInit(struct lContext * ctx, struct lScalar * scalar, enum lType val_type, size_t val_length);

L_RUNTIME_UNNESTED void lScalarRead  (struct lContext * ctx, struct lScalar * scalar, enum lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * result);
L_RUNTIME_UNNESTED void lScalarWrite (struct lContext * ctx, struct lScalar * scalar, enum lReduction reduction, enum lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * value);

L_RUNTIME_UNNESTED void lScalarEnterPhase (struct lScalar * scalar, enum lType val_type, size_t val_length, enum lPhase phase);

// L_RUNTIME_UNNESTED void lSetInitBoundary(struct lContext * ctx, struct lSet * set, enum lElementType type, const char * boundary_name);
L_RUNTIME_UNNESTED size_t lSetSize(struct lContext * ctx, struct lSet * set);

L_RUNTIME_UNNESTED void lKernelRun (struct lContext * ctx,  struct lSet * set, enum lElementType typ, int id, void (*kernel)(struct lkContext));
/*kernels have the signature and structure: 
void kernel_name(lkContext ctx_) {
	lkContext * ctx = &ctx_;
	lkElement e;
	if(lkGetActiveElement(ctx,&e)) {
	  <...>
	}
}*/

L_RUNTIME_UNNESTED void lPrintBegin (struct lContext * ctx);
L_RUNTIME_UNNESTED void lPrintEnd   (struct lContext * ctx);
L_RUNTIME_UNNESTED void lPrintValue (struct lContext * ctx, enum lType typ, size_t r, size_t c, const void * value);
L_RUNTIME_UNNESTED double lWallTime (struct lContext * ctx); //counter, time reported in seconds

//Nested Runtime Calls
L_RUNTIME_NESTED void lkPrintBegin   (struct lkContext * ctx);
L_RUNTIME_NESTED void lkPrintEnd     (struct lkContext * ctx);
L_RUNTIME_NESTED void lkPrintValue   (struct lkContext * ctx, enum lType typ, size_t r, size_t c, const void * value);
L_RUNTIME_NESTED void lkPrintElement (struct lkContext * ctx, enum lElementType typ, struct lkElement);

L_RUNTIME_NESTED struct lkFields  * lkGetFields        (struct lkContext * ctx);
L_RUNTIME_NESTED struct lkScalars * lkGetScalars       (struct lkContext * ctx);
L_RUNTIME_NESTED BOOL               lkGetActiveElement (struct lkContext * ctx, struct lkElement * e);

L_RUNTIME_NESTED void lkFieldRead  (struct lkField * scalar, struct lkElement e, enum lType element_type, size_t element_length, size_t val_offset, size_t val_length,  void * result);
L_RUNTIME_NESTED void lkFieldWrite (struct lkField * scalar, struct lkElement e, enum lReduction reduction, enum lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * value);

L_RUNTIME_NESTED void lkScalarRead  (struct lkContext * ctx, struct lkScalar * scalar, enum lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * result);
L_RUNTIME_NESTED void lkScalarWrite (struct lkContext * ctx, struct lkScalar * scalar, enum lReduction reduction, enum lType element_type, size_t element_length, size_t val_offset, size_t val_length, void * value);

L_RUNTIME_NESTED size_t lkSetSize (struct lkContext * ctx, struct lkSet * set);

L_RUNTIME_NESTED void lkSetGetIterator (struct lkContext * ctx, struct lkSet * set, enum lElementType typ, struct lkIterator * iterator);
L_RUNTIME_NESTED BOOL lkIteratorNext   (struct lkContext * ctx, struct lkIterator * iterator, enum lElementType typ, struct lkElement * e, int * lbl);

// L_RUNTIME_NESTED BOOL operator==(struct lkElement a, struct lkElement b);
// L_RUNTIME_NESTED static inline BOOL operator!=(struct lkElement a, struct lkElement b) { return !(a == b); }

//now all the topological functions ....

L_RUNTIME_UNNESTED void lVerticesOfMesh (struct lContext * ctx, struct lSet * set);
L_RUNTIME_UNNESTED void lEdgesOfMesh    (struct lContext * ctx, struct lSet * set);
L_RUNTIME_UNNESTED void lFacesOfMesh    (struct lContext * ctx, struct lSet * set);
L_RUNTIME_UNNESTED void lCellsOfMesh    (struct lContext * ctx, struct lSet * set);

L_RUNTIME_NESTED void lkVerticesOfVertex (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkVerticesOfEdge   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkVerticesOfFace   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkVerticesOfCell   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);

L_RUNTIME_NESTED void lkCellsOfVertex (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkCellsOfEdge   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkCellsOfFace   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkCellsOfCell   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);

L_RUNTIME_NESTED void lkEdgesOfVertex (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkEdgesOfFace   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkEdgesOfCell   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);

L_RUNTIME_NESTED void lkEdgesOfFaceCCW (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkEdgesOfFaceCW  (struct lkContext * ctx, struct lkElement c, struct lkSet * set);

L_RUNTIME_NESTED void lkFacesOfVertex (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkFacesOfEdge   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkFacesOfCell   (struct lkContext * ctx, struct lkElement c, struct lkSet * set);

L_RUNTIME_NESTED void lkFacesOfEdgeCCW (struct lkContext * ctx, struct lkElement c, struct lkSet * set);
L_RUNTIME_NESTED void lkFacesOfEdgeCW  (struct lkContext * ctx, struct lkElement c, struct lkSet * set);


L_RUNTIME_NESTED struct lkElement lkHeadOfEdge (struct lkContext * ctx, struct lkElement c);
struct lkElement lkTailOfEdge (struct lkContext * ctx, struct lkElement c);

L_RUNTIME_NESTED struct lkElement lkHeadOfFace (struct lkContext * ctx, struct lkElement c);
L_RUNTIME_NESTED struct lkElement lkTailOfFace (struct lkContext * ctx, struct lkElement c);

L_RUNTIME_NESTED struct lkElement lkFlipEdge (struct lkContext * ctx, struct lkElement c);
L_RUNTIME_NESTED struct lkElement lkFlipFace (struct lkContext * ctx, struct lkElement c);

L_RUNTIME_NESTED struct lkElement lkTowardsEdge (struct lkContext * ctx, struct lkElement e, struct lkElement v);
L_RUNTIME_NESTED struct lkElement lkTowardsFace (struct lkContext * ctx, struct lkElement e, struct lkElement v);

L_RUNTIME_NESTED int lkIDOfVertex (struct lkContext * ctx, struct lkElement e);
L_RUNTIME_NESTED int lkIDOfEdge   (struct lkContext * ctx, struct lkElement e);
L_RUNTIME_NESTED int lkIDOfFace   (struct lkContext * ctx, struct lkElement e);
L_RUNTIME_NESTED int lkIDOfCell   (struct lkContext * ctx, struct lkElement e);


L_RUNTIME_NESTED struct lkElement lkVertexOfCellWithLabel (struct lkContext * ctx, struct lkElement e, int l);
L_RUNTIME_NESTED struct lkElement lkEdgeOfCellWithLabel   (struct lkContext * ctx, struct lkElement e, int l);
L_RUNTIME_NESTED struct lkElement lkFaceOfCellWithLabel   (struct lkContext * ctx, struct lkElement e, int l);
L_RUNTIME_NESTED struct lkElement lkCellOfCellWithLabel   (struct lkContext * ctx, struct lkElement e, int l);


L_RUNTIME_NESTED struct lkElement lkVertexOfFaceWithLabel (struct lkContext * ctx, struct lkElement e, int l);
L_RUNTIME_NESTED struct lkElement lkEdgeOfFaceWithLabel   (struct lkContext * ctx, struct lkElement e, int l);
L_RUNTIME_NESTED struct lkElement lkCellOfFaceWithLabel   (struct lkContext * ctx, struct lkElement e, int l);

L_RUNTIME_NESTED struct lkElement lkCellOfEdgeWithLabel   (struct lkContext * ctx, struct lkElement e, int l);
L_RUNTIME_NESTED struct lkElement lkFaceOfEdgeWithLabel   (struct lkContext * ctx, struct lkElement e, int l);

//stencil generations functions
struct lsFunctionTable {
	L_RUNTIME_STENCIL void (*lsSetGetIterator)   (struct lsContext * ctx, struct lsSet * set, struct lsIterator * iterator);
	L_RUNTIME_STENCIL BOOL (*lsIteratorNext)     (struct lsContext * ctx, struct lsIterator * iterator, struct lsElement * e);
	L_RUNTIME_STENCIL void (*lsFieldAccess)      (struct lsContext * ctx, int field_id, enum lElementType type, struct lsElement * e, enum lPhase phase);
	L_RUNTIME_STENCIL void (*lsSetInitBoundary)  (struct lsContext * ctx, struct lsSet * set, enum lElementType type, const char * boundary_name);
	L_RUNTIME_STENCIL void (*lsGetActiveElement) (struct lsContext * ctx, struct lsElement * e);
	
	//stencil topology functions
	L_RUNTIME_STENCIL void (*lsVerticesOfMesh) (struct lsContext * ctx, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsEdgesOfMesh)    (struct lsContext * ctx, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsFacesOfMesh)    (struct lsContext * ctx, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsCellsOfMesh)    (struct lsContext * ctx, struct lsSet * set);
	
	L_RUNTIME_STENCIL void (*lsVerticesOfVertex) (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsVerticesOfEdge)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsVerticesOfFace)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsVerticesOfCell)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	
	L_RUNTIME_STENCIL void (*lsCellsOfVertex) (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsCellsOfEdge)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsCellsOfFace)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsCellsOfCell)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	
	L_RUNTIME_STENCIL void (*lsEdgesOfVertex) (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsEdgesOfFace)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsEdgesOfCell)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	
	L_RUNTIME_STENCIL void (*lsEdgesOfFaceCCW) (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsEdgesOfFaceCW)  (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	
	L_RUNTIME_STENCIL void (*lsFacesOfVertex) (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsFacesOfEdge)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsFacesOfCell)   (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	
	L_RUNTIME_STENCIL void (*lsFacesOfEdgeCCW) (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	L_RUNTIME_STENCIL void (*lsFacesOfEdgeCW)  (struct lsContext * ctx, struct lsElement * c, struct lsSet * set);
	
	
	L_RUNTIME_STENCIL void (*lsHeadOfEdge) (struct lsContext * ctx, struct lsElement * c, struct lsElement * result);
	L_RUNTIME_STENCIL void (*lsTailOfEdge) (struct lsContext * ctx, struct lsElement * c, struct lsElement * result);
	
 	L_RUNTIME_STENCIL void (*lsHeadOfFace) (struct lsContext * ctx, struct lsElement * c, struct lsElement * result);
	L_RUNTIME_STENCIL void (*lsTailOfFace) (struct lsContext * ctx, struct lsElement * c, struct lsElement * result);
	
	L_RUNTIME_STENCIL void (*lsFlipEdge) (struct lsContext * ctx, struct lsElement * c, struct lsElement * result);
	L_RUNTIME_STENCIL void (*lsFlipFace) (struct lsContext * ctx, struct lsElement * c, struct lsElement * result);
	
	L_RUNTIME_STENCIL void (*lsTowardsEdge)       (struct lsContext * ctx, struct lsElement * e, struct lsElement * v, struct lsElement * result);
	L_RUNTIME_STENCIL void (*lsTowardsFace)       (struct lsContext * ctx, struct lsElement * e, struct lsElement * v, struct lsElement * result);
	L_RUNTIME_STENCIL void (*lsSetExtractElement) (struct lsContext * ctx, struct lsSet * set, uint32_t elem, struct lsElement * result);
};

#endif
