/* Copyright 2015 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cfloat>
#include <cstdlib>

#include "legion.h"
#include "legion_c_util.h"

using namespace std;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor::AccessorType;

typedef CObjectWrapper::AccessorGeneric AccessorGeneric;

extern "C"
{
#include "legion_c.h"
#include "reductions_cpu.h"
}

#define ADD(x, y) ((x) + (y))
#define MUL(x, y) ((x) * (y))

// Pre-defined reduction operators
#define DECLARE_REDUCTION(REG, SRED, CLASS, T, T_N, U, APPLY_OP, FOLD_OP, N, ID) \
  class CLASS {                                                         \
  public:                                                               \
  typedef T_N LHS;                                                      \
  typedef T_N RHS;                                                      \
  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);       \
  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);      \
  static const T_N identity;                                            \
  };                                                                    \
                                                                        \
  const T_N CLASS::identity = { ID };                                   \
                                                                        \
  template <>                                                           \
  void CLASS::apply<true>(LHS &lhs, RHS rhs)                            \
  {                                                                     \
    for (int i = 0; i < N; ++i) {                                       \
      lhs.value[i] = APPLY_OP(lhs.value[i], rhs.value[i]);              \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<false>(LHS &lhs, RHS rhs)                           \
  {                                                                     \
    for (int i = 0; i < N; ++i) {                                       \
      U *target = (U *)&(lhs.value[i]);                                 \
      union { U as_U; T as_T; } oldval, newval;                         \
      do {                                                              \
        oldval.as_U = *target;                                          \
        newval.as_T = APPLY_OP(oldval.as_T, rhs.value[i]);              \
      } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<true>(RHS &rhs1, RHS rhs2)                           \
  {                                                                     \
    for (int i = 0; i < N; ++i) {                                       \
      rhs1.value[i] = FOLD_OP(rhs1.value[i], rhs2.value[i]);            \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<false>(RHS &rhs1, RHS rhs2)                          \
  {                                                                     \
    for (int i = 0; i < N; ++i) {                                       \
      U *target = (U *)&(rhs1.value[i]);                                \
      union { U as_U; T as_T; } oldval, newval;                         \
      do {                                                              \
        oldval.as_U = *target;                                          \
        newval.as_T = FOLD_OP(oldval.as_T, rhs2.value[i]);              \
      } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
    }                                                                   \
  }                                                                     \
                                                                        \
  extern "C"                                                            \
  {                                                                     \
  void REG(legion_reduction_op_id_t redop)                              \
  {                                                                     \
    HighLevelRuntime::register_reduction_op<CLASS>(redop);              \
  }                                                                     \
  void SRED(legion_accessor_generic_t accessor_,                        \
           legion_ptr_t ptr_, T_N value)                                \
  {                                                                     \
    AccessorGeneric* accessor = CObjectWrapper::unwrap(accessor_);      \
    ptr_t ptr = CObjectWrapper::unwrap(ptr_);                           \
    assert(accessor->typeify<T>().can_convert<ReductionFold<CLASS> >()); \
    accessor->typeify<T>().convert<ReductionFold<CLASS> >().reduce(ptr, value); \
  }                                                                     \
  }                                                                     \

// declare plus reductions on scalars
DECLARE_REDUCTION(register_reduction_plus_float,
                  safe_reduce_plus_float,
                  PlusOpFloat, float, float_1, int, ADD, ADD, 1, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double,
                  safe_reduce_plus_double,
                  PlusOpDouble, double, double_1, size_t, ADD, ADD, 1, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32,
                  safe_reduce_plus_int32,
                  PlusOpInt, int, int_1, int, ADD, ADD, 1, 0)

// declare plus reductions on vectors
DECLARE_REDUCTION(register_reduction_plus_float_vec2,
                  safe_reduce_plus_float_vec2,
                  PlusOpFloatVec2, float, float_2, int, ADD, ADD, 2, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_vec2,
                  safe_reduce_plus_double_vec2,
                  PlusOpDoubleVec2, double, double_2, size_t, ADD, ADD, 2, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_vec2,
                  safe_reduce_plus_int32_vec2,
                  PlusOpIntVec2, int, int_2, int, ADD, ADD, 2, 0)
DECLARE_REDUCTION(register_reduction_plus_float_vec3,
                  safe_reduce_plus_float_vec3,
                  PlusOpFloatVec3, float, float_3, int, ADD, ADD, 3, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_vec3,
                  safe_reduce_plus_double_vec3,
                  PlusOpDoubleVec3, double, double_3, size_t, ADD, ADD, 3, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_vec3,
                  safe_reduce_plus_int32_vec3,
                  PlusOpIntVec3, int, int_3, int, ADD, ADD, 3, 0)
DECLARE_REDUCTION(register_reduction_plus_float_vec4,
                  safe_reduce_plus_float_vec4,
                  PlusOpFloatVec4, float, float_4, int, ADD, ADD, 4, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_vec4,
                  safe_reduce_plus_double_vec4,
                  PlusOpDoubleVec4, double, double_4, size_t, ADD, ADD, 4, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_vec4,
                  safe_reduce_plus_int32_vec4,
                  PlusOpIntVec4, int, int_4, int, ADD, ADD, 4, 0)

// declare plus reductions on vectors
DECLARE_REDUCTION(register_reduction_plus_float_mat2x2,
                  safe_reduce_plus_float_mat2x2,
                  PlusOpFloatMat2x2, float, float_4, int, ADD, ADD, 4, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat2x2,
                  safe_reduce_plus_double_mat2x2,
                  PlusOpDoubleMat2x2, double, double_4, size_t, ADD, ADD, 4, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat2x2,
                  safe_reduce_plus_int32_mat2x2,
                  PlusOpIntMat2x2, int, int_4, int, ADD, ADD, 4, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat2x3,
                  safe_reduce_plus_float_mat2x3,
                  PlusOpFloatMat2x3, float, float_6, int, ADD, ADD, 6,  0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat2x3,
                  safe_reduce_plus_double_mat2x3,
                  PlusOpDoubleMat2x3, double, double_6, size_t, ADD, ADD, 6, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat2x3,
                  safe_reduce_plus_int32_mat2x3,
                  PlusOpIntMat2x3, int, int_6, int, ADD, ADD, 6, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat2x4,
                  safe_reduce_plus_float_mat2x4,
                  PlusOpFloatMat2x4, float, float_8, int, ADD, ADD, 8, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat2x4,
                  safe_reduce_plus_double_mat2x4,
                  PlusOpDoubleMat2x4, double, double_8, size_t, ADD, ADD, 8, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat2x4,
                  safe_reduce_plus_int32_mat2x4,
                  PlusOpIntMat2x4, int, int_8, int, ADD, ADD, 8, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat3x2,
                  safe_reduce_plus_float_mat3x2,
                  PlusOpFloatMat3x2, float, float_6, int, ADD, ADD, 6, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat3x2,
                  safe_reduce_plus_double_mat3x2,
                  PlusOpDoubleMat3x2, double, double_6, size_t, ADD, ADD, 6, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat3x2,
                  safe_reduce_plus_int32_mat3x2,
                  PlusOpIntMat3x2, int, int_6, int, ADD, ADD, 6, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat3x3,
                  safe_reduce_plus_float_mat3x3,
                  PlusOpFloatMat3x3, float, float_9, int, ADD, ADD, 9,  0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat3x3,
                  safe_reduce_plus_double_mat3x3,
                  PlusOpDoubleMat3x3, double, double_9, size_t, ADD, ADD, 9, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat3x3,
                  safe_reduce_plus_int32_mat3x3,
                  PlusOpIntMat3x3, int, int_9, int, ADD, ADD, 9, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat3x4,
                  safe_reduce_plus_float_mat3x4,
                  PlusOpFloatMat3x4, float, float_12, int, ADD, ADD, 12, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat3x4,
                  safe_reduce_plus_double_mat3x4,
                  PlusOpDoubleMat3x4, double, double_12, size_t, ADD, ADD, 12, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat3x4,
                  safe_reduce_plus_int32_mat3x4,
                  PlusOpIntMat3x4, int, int_12, int, ADD, ADD, 12, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat4x2,
                  safe_reduce_plus_float_mat4x2,
                  PlusOpFloatMat4x2, float, float_8, int, ADD, ADD, 8, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat4x2,
                  safe_reduce_plus_double_mat4x2,
                  PlusOpDoubleMat4x2, double, double_8, size_t, ADD, ADD, 8, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat4x2,
                  safe_reduce_plus_int32_mat4x2,
                  PlusOpIntMat4x2, int, int_8, int, ADD, ADD, 8, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat4x3,
                  safe_reduce_plus_float_mat4x3,
                  PlusOpFloatMat4x3, float, float_12, int, ADD, ADD, 12,  0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat4x3,
                  safe_reduce_plus_double_mat4x3,
                  PlusOpDoubleMat4x3, double, double_12, size_t, ADD, ADD, 12, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat4x3,
                  safe_reduce_plus_int32_mat4x3,
                  PlusOpIntMat4x3, int, int_12, int, ADD, ADD, 12, 0)
DECLARE_REDUCTION(register_reduction_plus_float_mat4x4,
                  safe_reduce_plus_float_mat4x4,
                  PlusOpFloatMat4x4, float, float_16, int, ADD, ADD, 16, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double_mat4x4,
                  safe_reduce_plus_double_mat4x4,
                  PlusOpDoubleMat4x4, double, double_16, size_t, ADD, ADD, 16, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32_mat4x4,
                  safe_reduce_plus_int32_mat4x4,
                  PlusOpIntMat4x4, int, int_16, int, ADD, ADD, 16, 0)

// declare times reductions on scalars
DECLARE_REDUCTION(register_reduction_times_float,
                  safe_reduce_times_float,
                  TimesOpFloat, float, float_1, int, MUL, MUL, 1, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double,
                  safe_reduce_times_double,
                  TimesOpDouble, double, double_1, size_t, MUL, MUL, 1, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32,
                  safe_reduce_times_int32,
                  TimesOpInt, int, int_1, int, MUL, MUL, 1, 0)

// declare times reductions on vectors
DECLARE_REDUCTION(register_reduction_times_float_vec2,
                  safe_reduce_times_float_vec2,
                  TimesOpFloatVec2, float, float_2, int, MUL, MUL, 2, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_vec2,
                  safe_reduce_times_double_vec2,
                  TimesOpDoubleVec2, double, double_2, size_t, MUL, MUL, 2, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_vec2,
                  safe_reduce_times_int32_vec2,
                  TimesOpIntVec2, int, int_2, int, MUL, MUL, 2, 0)
DECLARE_REDUCTION(register_reduction_times_float_vec3,
                  safe_reduce_times_float_vec3,
                  TimesOpFloatVec3, float, float_3, int, MUL, MUL, 3, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_vec3,
                  safe_reduce_times_double_vec3,
                  TimesOpDoubleVec3, double, double_3, size_t, MUL, MUL, 3, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_vec3,
                  safe_reduce_times_int32_vec3,
                  TimesOpIntVec3, int, int_3, int, MUL, MUL, 3, 0)
DECLARE_REDUCTION(register_reduction_times_float_vec4,
                  safe_reduce_times_float_vec4,
                  TimesOpFloatVec4, float, float_4, int, MUL, MUL, 4, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_vec4,
                  safe_reduce_times_double_vec4,
                  TimesOpDoubleVec4, double, double_4, size_t, MUL, MUL, 4, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_vec4,
                  safe_reduce_times_int32_vec4,
                  TimesOpIntVec4, int, int_4, int, MUL, MUL, 4, 0)

// declare times reductions on vectors
DECLARE_REDUCTION(register_reduction_times_float_mat2x2,
                  safe_reduce_times_float_mat2x2,
                  TimesOpFloatMat2x2, float, float_4, int, MUL, MUL, 4, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat2x2,
                  safe_reduce_times_double_mat2x2,
                  TimesOpDoubleMat2x2, double, double_4, size_t, MUL, MUL, 4, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat2x2,
                  safe_reduce_times_int32_mat2x2,
                  TimesOpIntMat2x2, int, int_4, int, MUL, MUL, 4, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat2x3,
                  safe_reduce_times_float_mat2x3,
                  TimesOpFloatMat2x3, float, float_6, int, MUL, MUL, 6,  0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat2x3,
                  safe_reduce_times_double_mat2x3,
                  TimesOpDoubleMat2x3, double, double_6, size_t, MUL, MUL, 6, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat2x3,
                  safe_reduce_times_int32_mat2x3,
                  TimesOpIntMat2x3, int, int_6, int, MUL, MUL, 6, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat2x4,
                  safe_reduce_times_float_mat2x4,
                  TimesOpFloatMat2x4, float, float_8, int, MUL, MUL, 8, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat2x4,
                  safe_reduce_times_double_mat2x4,
                  TimesOpDoubleMat2x4, double, double_8, size_t, MUL, MUL, 8, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat2x4,
                  safe_reduce_times_int32_mat2x4,
                  TimesOpIntMat2x4, int, int_8, int, MUL, MUL, 8, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat3x2,
                  safe_reduce_times_float_mat3x2,
                  TimesOpFloatMat3x2, float, float_6, int, MUL, MUL, 6, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat3x2,
                  safe_reduce_times_double_mat3x2,
                  TimesOpDoubleMat3x2, double, double_6, size_t, MUL, MUL, 6, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat3x2,
                  safe_reduce_times_int32_mat3x2,
                  TimesOpIntMat3x2, int, int_6, int, MUL, MUL, 6, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat3x3,
                  safe_reduce_times_float_mat3x3,
                  TimesOpFloatMat3x3, float, float_9, int, MUL, MUL, 9,  0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat3x3,
                  safe_reduce_times_double_mat3x3,
                  TimesOpDoubleMat3x3, double, double_9, size_t, MUL, MUL, 9, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat3x3,
                  safe_reduce_times_int32_mat3x3,
                  TimesOpIntMat3x3, int, int_9, int, MUL, MUL, 9, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat3x4,
                  safe_reduce_times_float_mat3x4,
                  TimesOpFloatMat3x4, float, float_12, int, MUL, MUL, 12, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat3x4,
                  safe_reduce_times_double_mat3x4,
                  TimesOpDoubleMat3x4, double, double_12, size_t, MUL, MUL, 12, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat3x4,
                  safe_reduce_times_int32_mat3x4,
                  TimesOpIntMat3x4, int, int_12, int, MUL, MUL, 12, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat4x2,
                  safe_reduce_times_float_mat4x2,
                  TimesOpFloatMat4x2, float, float_8, int, MUL, MUL, 8, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat4x2,
                  safe_reduce_times_double_mat4x2,
                  TimesOpDoubleMat4x2, double, double_8, size_t, MUL, MUL, 8, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat4x2,
                  safe_reduce_times_int32_mat4x2,
                  TimesOpIntMat4x2, int, int_8, int, MUL, MUL, 8, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat4x3,
                  safe_reduce_times_float_mat4x3,
                  TimesOpFloatMat4x3, float, float_12, int, MUL, MUL, 12,  0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat4x3,
                  safe_reduce_times_double_mat4x3,
                  TimesOpDoubleMat4x3, double, double_12, size_t, MUL, MUL, 12, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat4x3,
                  safe_reduce_times_int32_mat4x3,
                  TimesOpIntMat4x3, int, int_12, int, MUL, MUL, 12, 0)
DECLARE_REDUCTION(register_reduction_times_float_mat4x4,
                  safe_reduce_times_float_mat4x4,
                  TimesOpFloatMat4x4, float, float_16, int, MUL, MUL, 16, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double_mat4x4,
                  safe_reduce_times_double_mat4x4,
                  TimesOpDoubleMat4x4, double, double_16, size_t, MUL, MUL, 16, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32_mat4x4,
                  safe_reduce_times_int32_mat4x4,
                  TimesOpIntMat4x4, int, int_16, int, MUL, MUL, 16, 0)

// declare max reductions on scalars
DECLARE_REDUCTION(register_reduction_max_float,
                  safe_reduce_max_float,
                  MaxOpFloat, float, float_1, int, max, max, 1, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double,
                  safe_reduce_max_double,
                  MaxOpDouble, double, double_1, size_t, max, max, 1, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32,
                  safe_reduce_max_int32,
                  MaxOpInt, int, int_1, int, max, max, 1, INT_MAX)

// declare max reductions on vectors
DECLARE_REDUCTION(register_reduction_max_float_vec2,
                  safe_reduce_max_float_vec2,
                  MaxOpFloatVec2, float, float_2, int, max, max, 2, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_vec2,
                  safe_reduce_max_double_vec2,
                  MaxOpDoubleVec2, double, double_2, size_t, max, max, 2, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_vec2,
                  safe_reduce_max_int32_vec2,
                  MaxOpIntVec2, int, int_2, int, max, max, 2, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_vec3,
                  safe_reduce_max_float_vec3,
                  MaxOpFloatVec3, float, float_3, int, max, max, 3, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_vec3,
                  safe_reduce_max_double_vec3,
                  MaxOpDoubleVec3, double, double_3, size_t, max, max, 3, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_vec3,
                  safe_reduce_max_int32_vec3,
                  MaxOpIntVec3, int, int_3, int, max, max, 3, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_vec4,
                  safe_reduce_max_float_vec4,
                  MaxOpFloatVec4, float, float_4, int, max, max, 4, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_vec4,
                  safe_reduce_max_double_vec4,
                  MaxOpDoubleVec4, double, double_4, size_t, max, max, 4, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_vec4,
                  safe_reduce_max_int32_vec4,
                  MaxOpIntVec4, int, int_4, int, max, max, 4, INT_MAX)

// declare max reductions on vectors
DECLARE_REDUCTION(register_reduction_max_float_mat2x2,
                  safe_reduce_max_float_mat2x2,
                  MaxOpFloatMat2x2, float, float_4, int, max, max, 4, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat2x2,
                  safe_reduce_max_double_mat2x2,
                  MaxOpDoubleMat2x2, double, double_4, size_t, max, max, 4, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat2x2,
                  safe_reduce_max_int32_mat2x2,
                  MaxOpIntMat2x2, int, int_4, int, max, max, 4, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat2x3,
                  safe_reduce_max_float_mat2x3,
                  MaxOpFloatMat2x3, float, float_6, int, max, max, 6,  FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat2x3,
                  safe_reduce_max_double_mat2x3,
                  MaxOpDoubleMat2x3, double, double_6, size_t, max, max, 6, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat2x3,
                  safe_reduce_max_int32_mat2x3,
                  MaxOpIntMat2x3, int, int_6, int, max, max, 6, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat2x4,
                  safe_reduce_max_float_mat2x4,
                  MaxOpFloatMat2x4, float, float_8, int, max, max, 8, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat2x4,
                  safe_reduce_max_double_mat2x4,
                  MaxOpDoubleMat2x4, double, double_8, size_t, max, max, 8, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat2x4,
                  safe_reduce_max_int32_mat2x4,
                  MaxOpIntMat2x4, int, int_8, int, max, max, 8, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat3x2,
                  safe_reduce_max_float_mat3x2,
                  MaxOpFloatMat3x2, float, float_6, int, max, max, 6, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat3x2,
                  safe_reduce_max_double_mat3x2,
                  MaxOpDoubleMat3x2, double, double_6, size_t, max, max, 6, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat3x2,
                  safe_reduce_max_int32_mat3x2,
                  MaxOpIntMat3x2, int, int_6, int, max, max, 6, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat3x3,
                  safe_reduce_max_float_mat3x3,
                  MaxOpFloatMat3x3, float, float_9, int, max, max, 9,  FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat3x3,
                  safe_reduce_max_double_mat3x3,
                  MaxOpDoubleMat3x3, double, double_9, size_t, max, max, 9, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat3x3,
                  safe_reduce_max_int32_mat3x3,
                  MaxOpIntMat3x3, int, int_9, int, max, max, 9, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat3x4,
                  safe_reduce_max_float_mat3x4,
                  MaxOpFloatMat3x4, float, float_12, int, max, max, 12, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat3x4,
                  safe_reduce_max_double_mat3x4,
                  MaxOpDoubleMat3x4, double, double_12, size_t, max, max, 12, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat3x4,
                  safe_reduce_max_int32_mat3x4,
                  MaxOpIntMat3x4, int, int_12, int, max, max, 12, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat4x2,
                  safe_reduce_max_float_mat4x2,
                  MaxOpFloatMat4x2, float, float_8, int, max, max, 8, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat4x2,
                  safe_reduce_max_double_mat4x2,
                  MaxOpDoubleMat4x2, double, double_8, size_t, max, max, 8, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat4x2,
                  safe_reduce_max_int32_mat4x2,
                  MaxOpIntMat4x2, int, int_8, int, max, max, 8, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat4x3,
                  safe_reduce_max_float_mat4x3,
                  MaxOpFloatMat4x3, float, float_12, int, max, max, 12,  FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat4x3,
                  safe_reduce_max_double_mat4x3,
                  MaxOpDoubleMat4x3, double, double_12, size_t, max, max, 12, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat4x3,
                  safe_reduce_max_int32_mat4x3,
                  MaxOpIntMat4x3, int, int_12, int, max, max, 12, INT_MAX)
DECLARE_REDUCTION(register_reduction_max_float_mat4x4,
                  safe_reduce_max_float_mat4x4,
                  MaxOpFloatMat4x4, float, float_16, int, max, max, 16, FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double_mat4x4,
                  safe_reduce_max_double_mat4x4,
                  MaxOpDoubleMat4x4, double, double_16, size_t, max, max, 16, DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32_mat4x4,
                  safe_reduce_max_int32_mat4x4,
                  MaxOpIntMat4x4, int, int_16, int, max, max, 16, INT_MAX)

// declare min reductions on scalars
DECLARE_REDUCTION(register_reduction_min_float,
                  safe_reduce_min_float,
                  MinOpFloat, float, float_1, int, min, min, 1, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double,
                  safe_reduce_min_double,
                  MinOpDouble, double, double_1, size_t, min, min, 1, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32,
                  safe_reduce_min_int32,
                  MinOpInt, int, int_1, int, min, min, 1, INT_MIN)

// declare min reductions on vectors
DECLARE_REDUCTION(register_reduction_min_float_vec2,
                  safe_reduce_min_float_vec2,
                  MinOpFloatVec2, float, float_2, int, min, min, 2, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_vec2,
                  safe_reduce_min_double_vec2,
                  MinOpDoubleVec2, double, double_2, size_t, min, min, 2, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_vec2,
                  safe_reduce_min_int32_vec2,
                  MinOpIntVec2, int, int_2, int, min, min, 2, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_vec3,
                  safe_reduce_min_float_vec3,
                  MinOpFloatVec3, float, float_3, int, min, min, 3, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_vec3,
                  safe_reduce_min_double_vec3,
                  MinOpDoubleVec3, double, double_3, size_t, min, min, 3, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_vec3,
                  safe_reduce_min_int32_vec3,
                  MinOpIntVec3, int, int_3, int, min, min, 3, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_vec4,
                  safe_reduce_min_float_vec4,
                  MinOpFloatVec4, float, float_4, int, min, min, 4, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_vec4,
                  safe_reduce_min_double_vec4,
                  MinOpDoubleVec4, double, double_4, size_t, min, min, 4, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_vec4,
                  safe_reduce_min_int32_vec4,
                  MinOpIntVec4, int, int_4, int, min, min, 4, INT_MIN)

// declare min reductions on vectors
DECLARE_REDUCTION(register_reduction_min_float_mat2x2,
                  safe_reduce_min_float_mat2x2,
                  MinOpFloatMat2x2, float, float_4, int, min, min, 4, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat2x2,
                  safe_reduce_min_double_mat2x2,
                  MinOpDoubleMat2x2, double, double_4, size_t, min, min, 4, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat2x2,
                  safe_reduce_min_int32_mat2x2,
                  MinOpIntMat2x2, int, int_4, int, min, min, 4, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat2x3,
                  safe_reduce_min_float_mat2x3,
                  MinOpFloatMat2x3, float, float_6, int, min, min, 6,  FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat2x3,
                  safe_reduce_min_double_mat2x3,
                  MinOpDoubleMat2x3, double, double_6, size_t, min, min, 6, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat2x3,
                  safe_reduce_min_int32_mat2x3,
                  MinOpIntMat2x3, int, int_6, int, min, min, 6, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat2x4,
                  safe_reduce_min_float_mat2x4,
                  MinOpFloatMat2x4, float, float_8, int, min, min, 8, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat2x4,
                  safe_reduce_min_double_mat2x4,
                  MinOpDoubleMat2x4, double, double_8, size_t, min, min, 8, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat2x4,
                  safe_reduce_min_int32_mat2x4,
                  MinOpIntMat2x4, int, int_8, int, min, min, 8, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat3x2,
                  safe_reduce_min_float_mat3x2,
                  MinOpFloatMat3x2, float, float_6, int, min, min, 6, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat3x2,
                  safe_reduce_min_double_mat3x2,
                  MinOpDoubleMat3x2, double, double_6, size_t, min, min, 6, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat3x2,
                  safe_reduce_min_int32_mat3x2,
                  MinOpIntMat3x2, int, int_6, int, min, min, 6, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat3x3,
                  safe_reduce_min_float_mat3x3,
                  MinOpFloatMat3x3, float, float_9, int, min, min, 9,  FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat3x3,
                  safe_reduce_min_double_mat3x3,
                  MinOpDoubleMat3x3, double, double_9, size_t, min, min, 9, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat3x3,
                  safe_reduce_min_int32_mat3x3,
                  MinOpIntMat3x3, int, int_9, int, min, min, 9, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat3x4,
                  safe_reduce_min_float_mat3x4,
                  MinOpFloatMat3x4, float, float_12, int, min, min, 12, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat3x4,
                  safe_reduce_min_double_mat3x4,
                  MinOpDoubleMat3x4, double, double_12, size_t, min, min, 12, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat3x4,
                  safe_reduce_min_int32_mat3x4,
                  MinOpIntMat3x4, int, int_12, int, min, min, 12, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat4x2,
                  safe_reduce_min_float_mat4x2,
                  MinOpFloatMat4x2, float, float_8, int, min, min, 8, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat4x2,
                  safe_reduce_min_double_mat4x2,
                  MinOpDoubleMat4x2, double, double_8, size_t, min, min, 8, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat4x2,
                  safe_reduce_min_int32_mat4x2,
                  MinOpIntMat4x2, int, int_8, int, min, min, 8, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat4x3,
                  safe_reduce_min_float_mat4x3,
                  MinOpFloatMat4x3, float, float_12, int, min, min, 12,  FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat4x3,
                  safe_reduce_min_double_mat4x3,
                  MinOpDoubleMat4x3, double, double_12, size_t, min, min, 12, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat4x3,
                  safe_reduce_min_int32_mat4x3,
                  MinOpIntMat4x3, int, int_12, int, min, min, 12, INT_MIN)
DECLARE_REDUCTION(register_reduction_min_float_mat4x4,
                  safe_reduce_min_float_mat4x4,
                  MinOpFloatMat4x4, float, float_16, int, min, min, 16, FLT_MIN)
DECLARE_REDUCTION(register_reduction_min_double_mat4x4,
                  safe_reduce_min_double_mat4x4,
                  MinOpDoubleMat4x4, double, double_16, size_t, min, min, 16, DBL_MIN)
DECLARE_REDUCTION(register_reduction_min_int32_mat4x4,
                  safe_reduce_min_int32_mat4x4,
                  MinOpIntMat4x4, int, int_16, int, min, min, 16, INT_MIN)
