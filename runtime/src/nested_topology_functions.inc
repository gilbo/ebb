//this is a template for defining the nested topology functions for a runtime
//assuming that it uses the data structures defined in nested_topology.h



//the macro NESTED_TOPOLOGY_GET_MESH(ctx) should be defined to
//get a pointer to the Mesh object given the nested context for the runtime
//the macro NESTED_TOPOLOGY_GET_SET(set) should be defined to return the lNestedSet
//object used to store the the result of a set operation


L_RUNTIME_NESTED
void lkVerticesOfVertex(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->vtov,c,
	                     &NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkVerticesOfEdge(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->etov,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkVerticesOfFace(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftov,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkVerticesOfCell(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctov,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkCellsOfVertex(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->vtoc,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkCellsOfEdge(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->etoc,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkCellsOfFace(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftoc,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkCellsOfCell(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctoc,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkEdgesOfVertex(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->vtoe,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkEdgesOfFace(lkContext * ctx, lkElement c, lkSet * set) {
	lkEdgesOfFaceCCW(ctx,c,set);
}
L_RUNTIME_NESTED
void lkEdgesOfCell(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctoe,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkEdgesOfFaceCCW(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftoe,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkEdgesOfFaceCW(lkContext * ctx, lkElement c, lkSet * set) {
	lkEdgesOfFaceCCW(ctx,lkFlipFace(ctx,c),set);
}
L_RUNTIME_NESTED
void lkFacesOfVertex(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->vtof,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkFacesOfEdge(lkContext * ctx, lkElement c, lkSet * set) {
	lkFacesOfEdgeCCW(ctx,c,set);
}
L_RUNTIME_NESTED
void lkFacesOfCell(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctof,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkFacesOfEdgeCCW(lkContext * ctx, lkElement c, lkSet * set) {
	lNestedElementsOfCRS(&NESTED_TOPOLOGY_GET_MESH(ctx)->etof,c,&NESTED_TOPOLOGY_GET_SET(set));
}
L_RUNTIME_NESTED
void lkFacesOfEdgeCW(lkContext * ctx, lkElement c, lkSet * set) {
	lkFacesOfEdgeCCW(ctx, lkFlipEdge(ctx,c), set);
}
L_RUNTIME_NESTED
lkElement lkHeadOfEdge(lkContext * ctx, lkElement c) {
	return lNestedElementOfCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->etov,c,0);
}
L_RUNTIME_NESTED
lkElement lkTailOfEdge(lkContext * ctx, lkElement c) {
	return lNestedElementOfCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->etov,c,1);
}
L_RUNTIME_NESTED
lkElement lkHeadOfFace(lkContext * ctx, lkElement c) {
	return lNestedElementOfCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftoc,c,0);
}
L_RUNTIME_NESTED
lkElement lkTailOfFace(lkContext * ctx, lkElement c) {
	return lNestedElementOfCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftoc,c,1);
}
L_RUNTIME_NESTED
lkElement lkFlipEdge(lkContext * ctx, lkElement c) {
	return lNestedFlip(c);
}
L_RUNTIME_NESTED
lkElement lkFlipFace(lkContext * ctx, lkElement c) {
	return lNestedFlip(c);
}
L_RUNTIME_NESTED
lkElement lkTowardsEdge(lkContext * ctx, lkElement e, lkElement v) {
	return lNestedTowardsCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->etov,e,v);
}
L_RUNTIME_NESTED
lkElement lkTowardsFace(lkContext * ctx, lkElement e, lkElement v) {
	return lNestedTowardsCRSConst(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftoc,e,v);
}

L_RUNTIME_NESTED
lkElement lkVertexOfCellWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctov,e,l);
}
L_RUNTIME_NESTED
lkElement lkEdgeOfCellWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctoe,e,l);
}
L_RUNTIME_NESTED
lkElement lkFaceOfCellWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctof,e,l);
}
L_RUNTIME_NESTED
lkElement lkCellOfCellWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->ctoc,e,l);
}

L_RUNTIME_NESTED
lkElement lkVertexOfFaceWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftov,e,l);
}
L_RUNTIME_NESTED
lkElement lkEdgeOfFaceWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftoe,e,l);
}
L_RUNTIME_NESTED
lkElement lkCellOfFaceWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSConstWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->ftoc,e,l);
}
L_RUNTIME_NESTED
lkElement lkCellOfEdgeWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->etoc,e,l);
}
L_RUNTIME_NESTED
lkElement lkFaceOfEdgeWithLabel(lkContext * ctx, lkElement e, int l) {
	return lNestedElementOfCRSWithLabel(&NESTED_TOPOLOGY_GET_MESH(ctx)->etof,e,l);
}

#undef NESTED_TOPOLOGY_GET_MESH
#undef NESTED_TOPOLOGY_GET_SET