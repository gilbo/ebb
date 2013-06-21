#ifndef MATRIX_H_
#define MATRIX_H_

#include "vector.h"

#define M_ALWAYS_INLINE __attribute__((always_inline))
#ifdef __CUDACC__
#define M_CLASS_MODIFIERS M_ALWAYS_INLINE __host__ __device__
#define M_MODIFIERS  static inline M_ALWAYS_INLINE __host__ __device__
#else
#define M_CLASS_MODIFIERS M_ALWAYS_INLINE
#define M_MODIFIERS  static inline M_ALWAYS_INLINE
#endif

template<size_t R, size_t C, typename scalar>
struct matrix {
    scalar data[R][C];
    M_CLASS_MODIFIERS scalar & operator()(size_t a, size_t b) {
        return data[a][b];
    }
    M_CLASS_MODIFIERS const scalar & operator()(size_t a, size_t b) const {
        return data[a][b];
    }
    M_CLASS_MODIFIERS matrix<R,C,scalar> operator-() const {
        matrix<R,C,scalar> m;
        for(size_t i = 0; i < R; i++)
            for(size_t j = 0; j < C; j++)
                m.data[i][j] = -data[i][j];
        return m;
    }
};

template<size_t R, size_t C, typename scalar>
M_MODIFIERS matrix<C,R,scalar> transpose(const matrix<R,C,scalar> & m) { 
	matrix<C,R,scalar> ret;
	for(size_t i = 0; i < R; i++)
		for(size_t j = 0; j < C; j++)
			ret.data[j][i] = m.data[i][j];
	return m;
}
    
template<size_t R, size_t C, typename scalar>
M_MODIFIERS matrix<R,C,scalar> identity_matrix() {
	matrix<R,C,scalar> m;
	for(size_t i = 0; i < R; i++) {
		for(size_t j = 0; j < C; j++) {
			if ( j == i ) {
				m.data[i][j] = 1;
			} else {
				m.data[i][j] = 0;
			}   
		} 
	}  
	return m;
}
    
template<size_t R,size_t C,typename scalar>
M_MODIFIERS  matrix<R,C,scalar> operator*(const scalar & lhs, const matrix<R,C,scalar> & rhs) {
    matrix<R,C,scalar> ret;
    for(size_t i = 0; i < R; i++)
        for(size_t j = 0; j < C; j++)
          ret.data[i][j] = lhs * rhs.data[i][j];
    return ret;
}



template<size_t R,size_t C,typename scalar>
M_MODIFIERS  matrix<R,C,scalar> operator/(const scalar & lhs, const matrix<R,C,scalar> & rhs) {
    matrix<R,C,scalar> ret;
    for(size_t i = 0; i < R; i++)
        for(size_t j = 0; j < C; j++)
          ret.data[i][j] = lhs / rhs.data[i][j];
    return ret;
}

template<size_t R,size_t C,typename scalar>
M_MODIFIERS  matrix<R,C,scalar> operator*(const matrix<R,C,scalar> & rhs,const scalar & lhs) {
    matrix<R,C,scalar> ret;
    for(size_t i = 0; i < R; i++)
        for(size_t j = 0; j < C; j++)
          ret.data[i][j] = rhs.data[i][j] * lhs;
    return ret;
}


template<size_t R,size_t C,typename scalar>
M_MODIFIERS  vec<R,scalar> operator*(const matrix<R,C,scalar> & rhs,const vec<C,scalar> & lhs) {
    vec<R,scalar> ret;
    for(size_t i = 0; i < R; i++) {
        ret[i] = rhs.data[i][0] * lhs[0];
        for(size_t j = 1; j < C; j++)
            ret[i] += rhs.data[i][j] * lhs[j];
    }
    return ret;
}

template<size_t R,size_t C,typename scalar>
M_MODIFIERS  vec<C,scalar> operator*(const vec<R,scalar> & lhs,const matrix<R,C,scalar> & rhs) {
    vec<C,scalar> ret;
    for(size_t i = 0; i < C; i++) {
        ret[i] = rhs.data[0][i] * lhs[0];
        for(size_t j = 1; j < R; j++)
            ret[i] += rhs.data[j][i] * lhs[j];
    }
    return ret;
}

template<size_t A,size_t B,size_t C,typename scalar>
M_MODIFIERS  matrix<A,C,scalar> operator*(const matrix<A,B,scalar> lhs, const matrix<B,C,scalar> & rhs) {
    matrix<A,C,scalar> m;
    for(size_t i = 0; i < A; i++) {
        for(size_t j = 0; j < C; j++) {
            m.data[i][j] = 0;
            for(size_t k = 0; k < B; k++) {
                m.data[i][j] += lhs.data[i][k] * rhs.data[k][j];
            }
        }
    }
    return m;
}

template<size_t R,size_t C,typename scalar>
M_MODIFIERS  matrix<R,C,scalar> operator/(const matrix<R,C,scalar> & rhs,const scalar & lhs) {
    matrix<R,C,scalar> ret;
    for(size_t i = 0; i < R; i++)
        for(size_t j = 0; j < C; j++)
          ret.data[i][j] = rhs.data[i][j] / lhs;
    return ret;
}

template<size_t R,size_t C,typename scalar>
M_MODIFIERS  matrix<R,C,scalar> operator+(const matrix<R,C,scalar> & lhs,const matrix<R,C,scalar> & rhs) {
    matrix<R,C,scalar> ret;
    for(size_t i = 0; i < R; i++)
        for(size_t j = 0; j < C; j++)
          ret.data[i][j] = rhs.data[i][j] + lhs.data[i][j];
    return ret;
}

template<size_t R,size_t C,typename scalar>
M_MODIFIERS  matrix<R,C,scalar> operator-(const matrix<R,C,scalar> & lhs,const matrix<R,C,scalar> & rhs) {
    matrix<R,C,scalar> ret;
    for(size_t i = 0; i < R; i++)
        for(size_t j = 0; j < C; j++)
          ret.data[i][j] = lhs.data[i][j] - rhs.data[i][j];
    return ret;
}

template<size_t R,size_t C,typename scalar>
M_MODIFIERS  vec<R,scalar> diag(const matrix<R,C,scalar> & rhs) {
    vec<R,scalar> ret;
    for(size_t i = 0; i < R; i++) {
        ret[i] = rhs.data[i][i] ;
    }     
    return ret;
}

template<size_t N,typename scalar>
M_MODIFIERS  matrix<N, N,scalar> diag(const vec<N,scalar> & lhs) {
    matrix<N,N,scalar> ret ;
    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j <= i; j++) {
        	if ( j == i ) {
        		ret.data[i][i] = lhs[i] ;
        	} else {
        		ret.data[i][j] = 0;
        		ret.data[j][i] = 0;
        	}   
        } 
    } 
    return ret ;    
}

template<size_t R,size_t C,typename scalar>
M_MODIFIERS  matrix<R, C,scalar> outer(const vec<R,scalar> & v1, const vec<C,scalar> & v2) {
    matrix<R,C,scalar> ret ;
    for(size_t i = 0; i < R; i++) {
        for(size_t j = 0; j < C; j++) {
            ret.data[i][j] = v1[i]*v2[j] ;
        }
    }
    return ret ;
}

template<typename T, size_t R, size_t C>
M_MODIFIERS vec<C,T> row(const matrix<R,C,T> & m, int r) {
    vec<C,T> ret;
    for(size_t i = 0; i < C; i++) {
    	ret.data[i] = m.data[r][i];
    }
    return ret;
}

template<typename T, size_t R, size_t C>
M_MODIFIERS vec<R,T> col(const matrix<R,C,T> & m, int c) {
	vec<R,T> ret;
	for(size_t i = 0; i < R; i++)
		ret.data[i] = m.data[i][c];
	return ret;
}

template<size_t R,size_t C, typename scalar> 
M_MODIFIERS bool operator==(const matrix<R,C,scalar> & lhs,const matrix<R,C,scalar> & rhs) {
	for(size_t i = 0; i < R; i++)
        for(size_t j = 0; j < C; j++)
			if(lhs.data[i][j] != rhs.data[i][j])
				return false;
	return true;
}

template<size_t R,size_t C, typename scalar> M_MODIFIERS
bool operator!=(const matrix<R,C,scalar> & lhs,const matrix<R,C,scalar> & rhs) {
	return (!(lhs == rhs)) ;
}



#undef M_MODIFIERS
#undef M_CLASS_MODIFIERS
#undef M_ALWAYS_INLINE

#endif /* MATRIX_H_ */
