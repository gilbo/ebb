#ifndef _RUNTIME_UTIL_H
#define _RUNTIME_UTIL_H
#include "common/liszt_runtime.h"
#include <limits.h>
#include <float.h>

#define L_ALWAYS_INLINE __attribute__((always_inline))


//functions that are useful to all runtimes
static size_t lUtilTypeSize(lType typ);
L_RUNTIME_ALL static void lUtilValueLocation(size_t index, lType element_type, size_t element_length, size_t val_offset, size_t val_length, size_t * byte_offset, size_t * size);
L_RUNTIME_ALL  void lUtilValueReduce(void * lhs_, lReduction op, lType typ, size_t n_elements, void * rhs_);



//implementation follows


L_RUNTIME_ALL L_ALWAYS_INLINE
static inline size_t lUtilTypeSize(lType typ) {
	switch(typ) {
		case L_INT: return sizeof(int);
		case L_BOOL: return sizeof(bool);
		case L_FLOAT: return sizeof(float);
		case L_DOUBLE: return sizeof(double);
		case L_STRING: return sizeof(const char *);
		default: return 0;
	}
}


L_RUNTIME_ALL L_ALWAYS_INLINE
void lUtilValueLocation(size_t index, lType element_type, size_t element_length, size_t val_offset, size_t val_length, size_t * byte_offset, size_t * size) {
	size_t scalar_size = lUtilTypeSize(element_type);
	*byte_offset = (index * element_length + val_offset) * scalar_size;
	*size = scalar_size * val_length;
}


L_RUNTIME_ALL L_ALWAYS_INLINE 
void lUtilValueReduce(void * lhs_, lReduction op, lType typ, size_t n_elements, void * rhs_) {
	char * lhs = (char *) lhs_;
	char * rhs = (char *) rhs_;
	size_t scalar_size = lUtilTypeSize(typ);
	for(int i = 0; i < n_elements; i++) {
		
		switch(typ) {
			case L_INT: {
				int * lhst = (int*) lhs;
				int * rhst = (int*) rhs;
				switch(op) {
					case L_ASSIGN:
						*lhst = *rhst;
						break;
					case L_PLUS:
						*lhst += *rhst;
						break;
					case L_MINUS:
						*lhst -= *rhst;
						break;
					case L_MULTIPLY:	
						*lhst *= *rhst;
						break;
					case L_DIVIDE:
						*lhst /= *rhst;
						break;
					case L_MIN:
						*lhst = (*lhst < *rhst) ? *lhst : *rhst;
						break;
					case L_MAX:
						*lhst = (*lhst > *rhst) ? *lhst : *rhst;
						break;
					case L_BOR:
						*lhst |= *rhst;
						break;
					case L_BAND:
						*lhst &= *rhst;
						break;
					case L_AND:
						*lhst = *lhst && *rhst;
						break;
					case L_OR:
						*lhst = *lhst || *rhst;
						break;
					case L_XOR:
						*lhst = *lhst ^ *rhst;
						break;
				}
			} break;
			case L_BOOL: {
				bool * lhst = (bool*) lhs;
				bool * rhst = (bool*) rhs;
				switch(op) {
					case L_ASSIGN:
						*lhst = *rhst;
						break;
					case L_AND:
						*lhst = *lhst && *rhst;
						break;
					case L_OR:
						*lhst = *lhst || *rhst;
						break;
					default:
						break;
				}
			} break;
			case L_FLOAT: {
				float * lhst = (float*) lhs;
				float * rhst = (float*) rhs;
				switch(op) {
					case L_ASSIGN:
						*lhst = *rhst;
						break;
					case L_PLUS:
						*lhst += *rhst;
						break;
					case L_MINUS:
						*lhst -= *rhst;
						break;
					case L_MULTIPLY:	
						*lhst *= *rhst;
						break;
					case L_DIVIDE:
						*lhst /= *rhst;
						break;
					case L_MIN:
						*lhst = (*lhst < *rhst) ? *lhst : *rhst;
						break;
					case L_MAX:
						*lhst = (*lhst > *rhst) ? *lhst : *rhst;
						break;
					default:
						break;
				}
			} break;
			case L_DOUBLE:  {
				double * lhst = (double*) lhs;
				double * rhst = (double*) rhs;
				switch(op) {
					case L_ASSIGN:
						*lhst = *rhst;
						break;
					case L_PLUS:
						*lhst += *rhst;
						break;
					case L_MINUS:
						*lhst -= *rhst;
						break;
					case L_MULTIPLY:	
						*lhst *= *rhst;
						break;
					case L_DIVIDE:
						*lhst /= *rhst;
						break;
					case L_MIN:
						*lhst = (*lhst < *rhst) ? *lhst : *rhst;
						break;
					case L_MAX:
						*lhst = (*lhst > *rhst) ? *lhst : *rhst;
						break;
					default:
						break;
				}
			} break;
			case L_STRING: {
				char ** lhst = (char**) lhs;
				char ** rhst = (char**) rhs;
				switch(op) {
					case L_ASSIGN:
						*lhst = *rhst;
						break;
					default:
						break;
				}
			} break;
		}
		
		lhs += scalar_size;
		rhs += scalar_size;
	}
}

struct lUtilValue {
	union {
		int i;
		bool b;
		char * c;
		float f;
		double d;
	};
};

L_RUNTIME_ALL L_ALWAYS_INLINE
void lUtilValueIdentity(lPhase phase, lType val_type, lUtilValue * v) {
	switch(phase) {
		default: break;
		case L_REDUCE_PLUS:
		    switch(val_type) {
		    	case L_INT: v->i = 0; break;
		    	case L_FLOAT: v->f = 0.f; break;
		    	case L_DOUBLE: v->d = 0.; break;
		    	default: break;
		    } break;
		case L_REDUCE_MULTIPLY:
		    switch(val_type) {
		    	case L_INT: v->i = 1; break;
		    	case L_FLOAT:  v->f = 1.f; break;
		    	case L_DOUBLE: v->d = 1.0; break;
		    	default: break;
		    } break;
		case L_REDUCE_MIN:
			switch(val_type) {
		    	case L_INT: v->i = INT_MAX; break;
		    	case L_FLOAT: v->f = FLT_MAX; break;
		    	case L_DOUBLE: v->d = DBL_MAX; break;
		    	default: break;
		    } break;
		case L_REDUCE_MAX:
			switch(val_type) {
		    	case L_INT: v->i = INT_MIN; break;
		    	case L_FLOAT: v->f = FLT_MIN; break;
		    	case L_DOUBLE: v->d = DBL_MIN; break;
		    	default: break;
		    } break;
		case L_REDUCE_BOR:
			switch(val_type) {
		    	case L_INT: v->i = 0; break;
		    	default: break;
		    } break;
		case L_REDUCE_BAND:
			switch(val_type) {
		    	case L_INT: v->i = ~0; break;
		    	default: break;
		    } break;
		case L_REDUCE_XOR:
			switch(val_type) {
		    	case L_INT: v->i = 0; break;
		    	default: break;
		    } break;
		case L_REDUCE_AND:
			switch(val_type) {
		    	case L_INT: v->i = 1; break;
		    	case L_BOOL: v->b = true; break;
		    	default: break;
		    } break;
		case L_REDUCE_OR:
			switch(val_type) {
		    	case L_INT: v->i = 0; break;
		    	case L_BOOL: v->b = false; break;
		    	default: break;
		    } break;
	}
}

L_RUNTIME_ALL L_ALWAYS_INLINE
lReduction lUtilPhaseToReduction(lPhase phase) {
	switch(phase) {
	case L_WRITE_ONLY: return L_ASSIGN;
	case L_REDUCE_PLUS: return L_PLUS;
	case L_REDUCE_MULTIPLY: return L_MULTIPLY;
	case L_REDUCE_MIN: return L_MIN;
	case L_REDUCE_MAX: return L_MAX;
	case L_REDUCE_BOR: return L_BOR;
	case L_REDUCE_BAND: return L_BAND;
	case L_REDUCE_XOR: return L_XOR;
	case L_REDUCE_AND: return L_AND;
	case L_REDUCE_OR: return L_OR;
	case L_READ_ONLY: return L_ASSIGN;
	case L_MODIFY: return L_ASSIGN;
	}
}

void lUtilParseProgramArguments(int argc, char ** argv, lProgramArguments * arguments) {
	arguments->mesh_file = argv[1];
	arguments->redirect_output_to_log = atoi(argv[2]) != 0;
}

#endif