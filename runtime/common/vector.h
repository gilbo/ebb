#ifndef __VECTOR_H
#define __VECTOR_H

#include<algorithm>
#include<cmath>

#define V_ALWAYS_INLINE __attribute__((always_inline))

#ifdef __CUDACC__
#define V_CLASS_MODIFIERS V_ALWAYS_INLINE __host__ __device__
#define V_MODIFIERS static inline V_ALWAYS_INLINE __host__ __device__
#else
#define V_CLASS_MODIFIERS V_ALWAYS_INLINE
#define V_MODIFIERS static inline V_ALWAYS_INLINE
#endif


//vec object must be POD to support its initialization syntax:
//   vec<3,int> a = {1,2,3};
//this which means there can be no constructors, among other things
template<size_t N, typename scalar> 
struct vec {
	scalar data[N];
	
	V_CLASS_MODIFIERS scalar & operator[](size_t i) { 
		return data[i];
	}
	V_CLASS_MODIFIERS const scalar & operator[](size_t i) const {
		return data[i];
	}
	V_CLASS_MODIFIERS vec<N,scalar> operator-() const {
		vec<N,scalar> ret;
		for(size_t i = 0; i < N; i++) 
			ret[i] = -data[i]; 
		return ret; 
	}
	template<typename scalar2> 
	V_CLASS_MODIFIERS operator vec<N,scalar2>() const {
		vec<N,scalar2> ret;
		for(size_t i = 0; i < N; i++)
			ret.data[i] = data[i];
		return ret;
	}
};


#define VEC_OP(name,function) \
template<size_t N, typename scalar> \
V_MODIFIERS vec<N,scalar> name (const vec<N,scalar> & lhs, const vec<N,scalar> & rhs) { \
	vec<N,scalar> ret; \
	for(size_t i = 0; i < N; i++) \
		ret.data[i] = function(lhs.data[i], rhs.data[i]); \
	return ret; \
} \
template<size_t N, typename scalar> \
V_MODIFIERS vec<N,scalar> name (const vec<N,scalar> & lhs, const scalar & rhs) { \
	vec<N,scalar> ret; \
	for(size_t i = 0; i < N; i++) \
		ret.data[i] = function(lhs.data[i], rhs); \
	return ret; \
} \
template<size_t N, typename scalar> \
V_MODIFIERS vec<N,scalar> name (const scalar & lhs, const vec<N,scalar> & rhs) { \
	vec<N,scalar> ret; \
	for(size_t i = 0; i < N; i++) \
		ret.data[i] = function(lhs, rhs.data[i]); \
	return ret; \
}

#define V_ADD(a,b) (a) + (b)
VEC_OP(operator +,V_ADD)
#undef V_ADD

#define V_SUB(a,b) (a) - (b)
VEC_OP(operator -,V_SUB)
#undef V_SUB

#define V_MULT(a,b) (a) * (b)
VEC_OP(operator *,V_MULT)
#undef V_MULT

#define V_DIV(a,b) (a) / (b)
VEC_OP(operator /,V_DIV)
#undef V_DIV

template<typename T>
V_MODIFIERS T vec_min_helper(const T & lhs, const T & rhs) {
	return (lhs < rhs) ? lhs : rhs;
}
template<typename T>
V_MODIFIERS T vec_max_helper(const T & lhs, const T & rhs) {
	return (lhs > rhs) ? lhs : rhs;
}
VEC_OP(min, vec_min_helper)
VEC_OP(max, vec_max_helper)

#undef VEC_OP

template<size_t N,typename scalar> 
V_MODIFIERS bool operator==(const vec<N,scalar> & lhs, const vec<N,scalar> & rhs) {
        for(size_t i = 0; i < N; i++)
            if(lhs[i] != rhs[i])
            	return false;
        return true;
}
template<size_t N,typename scalar> 
V_MODIFIERS bool operator!=(const vec<N,scalar> & lhs, const vec<N,scalar> & rhs) {
        return !(lhs == rhs);
}

template<typename scalar>
V_MODIFIERS vec<3,scalar> cross(const vec<3, scalar> & lhs,const vec<3, scalar> & rhs) {
	vec<3,scalar> ret;
    ret.data[0] = lhs.data[1]*rhs.data[2] - lhs.data[2]*rhs.data[1];
    ret.data[1] = lhs.data[2]*rhs.data[0] - lhs.data[0]*rhs.data[2];
    ret.data[2] = lhs.data[0]*rhs.data[1] - lhs.data[1]*rhs.data[0];
	return ret;
}

template<size_t N, typename scalar> 
V_MODIFIERS scalar dot(const vec<N,scalar> & lhs, const vec<N,scalar> & rhs) {
	scalar acc = lhs.data[0] * rhs.data[0];
	for(size_t i = 1; i < N; i++)
		acc += lhs.data[i] * rhs.data[i];
	return acc;
}

template<size_t N, typename scalar>
V_MODIFIERS vec<N,scalar> abs(const vec<N,scalar> & v) {
    vec<N,scalar> ret;
    for (size_t i = 0; i < N; i++) {
        ret.data[i] = std::abs(v.data[i]);
    }
    return ret;
}

template<size_t N, typename scalar>
V_MODIFIERS scalar length(const vec<N,scalar> & v) {
    scalar len = 0;
    for (size_t i = 0; i < N; i++) {
        len += v.data[i] * v.data[i];
    }
    return std::sqrt(len);
}


template<size_t N, typename scalar> 
V_MODIFIERS vec<N, scalar> normalize(const vec<N,scalar> & v) {
    return v / length(v);
}


#undef V_MODIFIERS
#undef V_CLASS_MODIFIERS
#undef V_ALWAYS_INLINE
#endif //header
