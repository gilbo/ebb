#ifndef _NESTED_TOPOLOGY_H
#define _NESTED_TOPOLOGY_H



#ifdef __CUDACC__
#define L_TOPOLOGY_MODIFIERS __host__ __device__
#else
#define L_TOPOLOGY_MODIFIERS
#endif

//simple definition of nested sets that can be used across many runtimes

struct lNestedSet {
	int dir; //1 or -1
	uint32_t * elements;
	uint32_t size;
};

struct lNestedIterator {
	lNestedSet * set;
	int32_t i;
};




L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
static inline size_t elementID(lkElement e) {
	return ~FLIP_DIRECTION_BIT & e.data;
}

L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
bool operator==(lkElement a, lkElement b) {
	return elementID(a) == elementID(b);
}



L_TOPOLOGY_MODIFIERS
void lNestedElementsOfCRS(CRS * crs, lkElement e, lNestedSet * set) {
	uint32_t id = elementID(e);
	set->size = crs->row_idx[id + 1] - crs->row_idx[id];
	if((e.data & FLIP_DIRECTION_BIT) == 0) {
		set->elements = crs->values + crs->row_idx[id];
		set->dir = 1;
	} else {
		set->elements = crs->values + crs->row_idx[id + 1] - 1;
		set->dir = -1;
	}
}

L_TOPOLOGY_MODIFIERS
void lNestedElementsOfCRSConst(CRSConst * crs, lkElement c, lNestedSet * set) {
	set->size = 2;
	set->dir = 1;
	set->elements = crs->values[elementID(c)];
}

L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
void lNestedSetGetIterator(lNestedSet * set, lNestedIterator * iterator) {
	iterator->set = set;
	iterator->i = 0;
}


L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
bool lNestedSetGetElement(lNestedSet * set, int idx, lkElement * e) {
	if(idx < set->size) {
		e->data = set->elements[idx * set->dir];
		e->data ^= (FLIP_DIRECTION_BIT & set->dir);
		return true;
	} else {
		return false;
	}
}

L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
bool lNestedIteratorNext(lNestedIterator * iterator, lkElement * e, int * lbl) {
	bool result = lNestedSetGetElement(iterator->set, iterator->i,e);
	if(lbl)
		*lbl = iterator->i;
	iterator->i++;
	return result;
}


L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
lkElement lNestedElementOfCRSConst(CRSConst * crs, lkElement c, int is_tail) {
	uint32_t idx = (c.data & FLIP_DIRECTION_BIT) >> FLIP_DIRECTION_SHIFT;
	lkElement r = { crs->values[elementID(c)][idx ^ is_tail] };
	return r;
}


L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
lkElement lNestedFlip(lkElement c) {
	lkElement r = { FLIP_DIRECTION_BIT ^ c.data };
	return r;
}

L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
lkElement lNestedTowardsCRSConst(CRSConst * crs, lkElement e, lkElement v) {
	if(elementID(lNestedElementOfCRSConst(crs,e,0)) == elementID(v))
		return e;
	else
		return lNestedFlip(e);
}

L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
lkElement lNestedElementOfCRSWithLabel(CRS * crs, lkElement e, int l) {
	lkElement r = { crs->values[crs->row_idx[elementID(e)] + l] };
	return r;
}

L_ALWAYS_INLINE L_TOPOLOGY_MODIFIERS
lkElement lNestedElementOfCRSConstWithLabel(CRSConst * crs, lkElement e, int l) {
	lkElement r = { crs->values[elementID(e)][l] };
	return r;
}


#endif