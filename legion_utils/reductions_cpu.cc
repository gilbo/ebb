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
#include <limits>

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


// FIELD DATA

// Pre-defined reduction operators
#define DECLARE_FIELD_REDUCTION(REG, SRED, SRED_DP, CLASS, T, T_N, U, APPLY_OP, FOLD_OP, N, ID) \
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
    accessor->typeify<T_N>().convert<ReductionFold<CLASS> >().reduce(ptr, value); \
  }                                                                     \
  void SRED_DP(legion_accessor_generic_t accessor_,                     \
               legion_domain_point_t dp_, T_N value)                    \
  {                                                                     \
    AccessorGeneric* accessor = CObjectWrapper::unwrap(accessor_);      \
    DomainPoint dp = CObjectWrapper::unwrap(dp_);                       \
    accessor->typeify<T_N>()/*.convert<ReductionFold<CLASS> >()*/.reduce<CLASS>(dp, value); \
  }                                                                     \
  }                                                                     \


// declare plus reductions on scalars
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float,
                  safe_reduce_plus_float, safe_reduce_domain_point_plus_float,
                  FieldPlusOpFloat, float, float_1, int, ADD, ADD, 1, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double,
                  safe_reduce_plus_double, safe_reduce_domain_point_plus_double,
                  FieldPlusOpDouble, double, double_1, size_t, ADD, ADD, 1, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32,
                  safe_reduce_plus_int32, safe_reduce_domain_point_plus_int32,
                  FieldPlusOpInt, int, int_1, int, ADD, ADD, 1, 0)

// declare plus reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_vec2,
                  safe_reduce_plus_float_vec2, safe_reduce_domain_point_plus_float_vec2,
                  FieldPlusOpFloatVec2, float, float_2, int, ADD, ADD, 2, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_vec2,
                  safe_reduce_plus_double_vec2, safe_reduce_domain_point_plus_double_vec2,
                  FieldPlusOpDoubleVec2, double, double_2, size_t, ADD, ADD, 2, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_vec2,
                  safe_reduce_plus_int32_vec2, safe_reduce_domain_point_plus_int32_vec2,
                  FieldPlusOpIntVec2, int, int_2, int, ADD, ADD, 2, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_vec3,
                  safe_reduce_plus_float_vec3, safe_reduce_domain_point_plus_float_vec3,
                  FieldPlusOpFloatVec3, float, float_3, int, ADD, ADD, 3, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_vec3,
                  safe_reduce_plus_double_vec3, safe_reduce_domain_point_plus_double_vec3,
                  FieldPlusOpDoubleVec3, double, double_3, size_t, ADD, ADD, 3, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_vec3,
                  safe_reduce_plus_int32_vec3, safe_reduce_domain_point_plus_int32_vec3,
                  FieldPlusOpIntVec3, int, int_3, int, ADD, ADD, 3, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_vec4,
                  safe_reduce_plus_float_vec4, safe_reduce_domain_point_plus_float_vec4,
                  FieldPlusOpFloatVec4, float, float_4, int, ADD, ADD, 4, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_vec4,
                  safe_reduce_plus_double_vec4, safe_reduce_domain_point_plus_double_vec4,
                  FieldPlusOpDoubleVec4, double, double_4, size_t, ADD, ADD, 4, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_vec4,
                  safe_reduce_plus_int32_vec4, safe_reduce_domain_point_plus_int32_vec4,
                  FieldPlusOpIntVec4, int, int_4, int, ADD, ADD, 4, 0)

// declare plus reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat2x2,
                  safe_reduce_plus_float_mat2x2, safe_reduce_domain_point_plus_float_mat2x2,
                  FieldPlusOpFloatMat2x2, float, float_4, int, ADD, ADD, 4, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat2x2,
                  safe_reduce_plus_double_mat2x2, safe_reduce_domain_point_plus_double_mat2x2,
                  FieldPlusOpDoubleMat2x2, double, double_4, size_t, ADD, ADD, 4, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat2x2,
                  safe_reduce_plus_int32_mat2x2, safe_reduce_domain_point_plus_int32_mat2x2,
                  FieldPlusOpIntMat2x2, int, int_4, int, ADD, ADD, 4, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat2x3,
                  safe_reduce_plus_float_mat2x3, safe_reduce_domain_point_plus_float_mat2x3,
                  FieldPlusOpFloatMat2x3, float, float_6, int, ADD, ADD, 6,  0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat2x3,
                  safe_reduce_plus_double_mat2x3, safe_reduce_domain_point_plus_double_mat2x3,
                  FieldPlusOpDoubleMat2x3, double, double_6, size_t, ADD, ADD, 6, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat2x3,
                  safe_reduce_plus_int32_mat2x3, safe_reduce_domain_point_plus_int32_mat2x3,
                  FieldPlusOpIntMat2x3, int, int_6, int, ADD, ADD, 6, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat2x4,
                  safe_reduce_plus_float_mat2x4, safe_reduce_domain_point_plus_float_mat2x4,
                  FieldPlusOpFloatMat2x4, float, float_8, int, ADD, ADD, 8, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat2x4,
                  safe_reduce_plus_double_mat2x4, safe_reduce_domain_point_plus_double_mat2x4,
                  FieldPlusOpDoubleMat2x4, double, double_8, size_t, ADD, ADD, 8, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat2x4,
                  safe_reduce_plus_int32_mat2x4, safe_reduce_domain_point_plus_int32_mat2x4,
                  FieldPlusOpIntMat2x4, int, int_8, int, ADD, ADD, 8, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat3x2,
                  safe_reduce_plus_float_mat3x2, safe_reduce_domain_point_plus_float_mat3x2,
                  FieldPlusOpFloatMat3x2, float, float_6, int, ADD, ADD, 6, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat3x2,
                  safe_reduce_plus_double_mat3x2, safe_reduce_domain_point_plus_double_mat3x2,
                  FieldPlusOpDoubleMat3x2, double, double_6, size_t, ADD, ADD, 6, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat3x2,
                  safe_reduce_plus_int32_mat3x2, safe_reduce_domain_point_plus_int32_mat3x2,
                  FieldPlusOpIntMat3x2, int, int_6, int, ADD, ADD, 6, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat3x3,
                  safe_reduce_plus_float_mat3x3, safe_reduce_domain_point_plus_float_mat3x3,
                  FieldPlusOpFloatMat3x3, float, float_9, int, ADD, ADD, 9,  0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat3x3,
                  safe_reduce_plus_double_mat3x3, safe_reduce_domain_point_plus_double_mat3x3,
                  FieldPlusOpDoubleMat3x3, double, double_9, size_t, ADD, ADD, 9, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat3x3,
                  safe_reduce_plus_int32_mat3x3, safe_reduce_domain_point_plus_int32_mat3x3,
                  FieldPlusOpIntMat3x3, int, int_9, int, ADD, ADD, 9, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat3x4,
                  safe_reduce_plus_float_mat3x4, safe_reduce_domain_point_plus_float_mat3x4,
                  FieldPlusOpFloatMat3x4, float, float_12, int, ADD, ADD, 12, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat3x4,
                  safe_reduce_plus_double_mat3x4, safe_reduce_domain_point_plus_double_mat3x4,
                  FieldPlusOpDoubleMat3x4, double, double_12, size_t, ADD, ADD, 12, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat3x4,
                  safe_reduce_plus_int32_mat3x4, safe_reduce_domain_point_plus_int32_mat3x4,
                  FieldPlusOpIntMat3x4, int, int_12, int, ADD, ADD, 12, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat4x2,
                  safe_reduce_plus_float_mat4x2, safe_reduce_domain_point_plus_float_mat4x2,
                  FieldPlusOpFloatMat4x2, float, float_8, int, ADD, ADD, 8, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat4x2,
                  safe_reduce_plus_double_mat4x2, safe_reduce_domain_point_plus_double_mat4x2,
                  FieldPlusOpDoubleMat4x2, double, double_8, size_t, ADD, ADD, 8, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat4x2,
                  safe_reduce_plus_int32_mat4x2, safe_reduce_domain_point_plus_int32_mat4x2,
                  FieldPlusOpIntMat4x2, int, int_8, int, ADD, ADD, 8, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat4x3,
                  safe_reduce_plus_float_mat4x3, safe_reduce_domain_point_plus_float_mat4x3,
                  FieldPlusOpFloatMat4x3, float, float_12, int, ADD, ADD, 12,  0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat4x3,
                  safe_reduce_plus_double_mat4x3, safe_reduce_domain_point_plus_double_mat4x3,
                  FieldPlusOpDoubleMat4x3, double, double_12, size_t, ADD, ADD, 12, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat4x3,
                  safe_reduce_plus_int32_mat4x3, safe_reduce_domain_point_plus_int32_mat4x3,
                  FieldPlusOpIntMat4x3, int, int_12, int, ADD, ADD, 12, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_float_mat4x4,
                  safe_reduce_plus_float_mat4x4, safe_reduce_domain_point_plus_float_mat4x4,
                  FieldPlusOpFloatMat4x4, float, float_16, int, ADD, ADD, 16, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_double_mat4x4,
                  safe_reduce_plus_double_mat4x4, safe_reduce_domain_point_plus_double_mat4x4,
                  FieldPlusOpDoubleMat4x4, double, double_16, size_t, ADD, ADD, 16, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_plus_int32_mat4x4,
                  safe_reduce_plus_int32_mat4x4, safe_reduce_domain_point_plus_int32_mat4x4,
                  FieldPlusOpIntMat4x4, int, int_16, int, ADD, ADD, 16, 0)

// declare times reductions on scalars
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float,
                  safe_reduce_times_float, safe_reduce_domain_point_times_float,
                  FieldTimesOpFloat, float, float_1, int, MUL, MUL, 1, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double,
                  safe_reduce_times_double, safe_reduce_domain_point_times_double,
                  FieldTimesOpDouble, double, double_1, size_t, MUL, MUL, 1, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32,
                  safe_reduce_times_int32, safe_reduce_domain_point_times_int32,
                  FieldTimesOpInt, int, int_1, int, MUL, MUL, 1, 0)

// declare times reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_vec2,
                  safe_reduce_times_float_vec2, safe_reduce_domain_point_times_float_vec2,
                  FieldTimesOpFloatVec2, float, float_2, int, MUL, MUL, 2, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_vec2,
                  safe_reduce_times_double_vec2, safe_reduce_domain_point_times_double_vec2,
                  FieldTimesOpDoubleVec2, double, double_2, size_t, MUL, MUL, 2, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_vec2,
                  safe_reduce_times_int32_vec2, safe_reduce_domain_point_times_int32_vec2,
                  FieldTimesOpIntVec2, int, int_2, int, MUL, MUL, 2, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_vec3,
                  safe_reduce_times_float_vec3, safe_reduce_domain_point_times_float_vec3,
                  FieldTimesOpFloatVec3, float, float_3, int, MUL, MUL, 3, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_vec3,
                  safe_reduce_times_double_vec3, safe_reduce_domain_point_times_double_vec3,
                  FieldTimesOpDoubleVec3, double, double_3, size_t, MUL, MUL, 3, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_vec3,
                  safe_reduce_times_int32_vec3, safe_reduce_domain_point_times_int32_vec3,
                  FieldTimesOpIntVec3, int, int_3, int, MUL, MUL, 3, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_vec4,
                  safe_reduce_times_float_vec4, safe_reduce_domain_point_times_float_vec4,
                  FieldTimesOpFloatVec4, float, float_4, int, MUL, MUL, 4, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_vec4,
                  safe_reduce_times_double_vec4, safe_reduce_domain_point_times_double_vec4,
                  FieldTimesOpDoubleVec4, double, double_4, size_t, MUL, MUL, 4, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_vec4,
                  safe_reduce_times_int32_vec4, safe_reduce_domain_point_times_int32_vec4,
                  FieldTimesOpIntVec4, int, int_4, int, MUL, MUL, 4, 0)

// declare times reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat2x2,
                  safe_reduce_times_float_mat2x2, safe_reduce_domain_point_times_float_mat2x2,
                  FieldTimesOpFloatMat2x2, float, float_4, int, MUL, MUL, 4, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat2x2,
                  safe_reduce_times_double_mat2x2, safe_reduce_domain_point_times_double_mat2x2,
                  FieldTimesOpDoubleMat2x2, double, double_4, size_t, MUL, MUL, 4, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat2x2,
                  safe_reduce_times_int32_mat2x2, safe_reduce_domain_point_times_int32_mat2x2,
                  FieldTimesOpIntMat2x2, int, int_4, int, MUL, MUL, 4, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat2x3,
                  safe_reduce_times_float_mat2x3, safe_reduce_domain_point_times_float_mat2x3,
                  FieldTimesOpFloatMat2x3, float, float_6, int, MUL, MUL, 6,  0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat2x3,
                  safe_reduce_times_double_mat2x3, safe_reduce_domain_point_times_double_mat2x3,
                  FieldTimesOpDoubleMat2x3, double, double_6, size_t, MUL, MUL, 6, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat2x3,
                  safe_reduce_times_int32_mat2x3, safe_reduce_domain_point_times_int32_mat2x3,
                  FieldTimesOpIntMat2x3, int, int_6, int, MUL, MUL, 6, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat2x4,
                  safe_reduce_times_float_mat2x4, safe_reduce_domain_point_times_float_mat2x4,
                  FieldTimesOpFloatMat2x4, float, float_8, int, MUL, MUL, 8, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat2x4,
                  safe_reduce_times_double_mat2x4, safe_reduce_domain_point_times_double_mat2x4,
                  FieldTimesOpDoubleMat2x4, double, double_8, size_t, MUL, MUL, 8, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat2x4,
                  safe_reduce_times_int32_mat2x4, safe_reduce_domain_point_times_int32_mat2x4,
                  FieldTimesOpIntMat2x4, int, int_8, int, MUL, MUL, 8, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat3x2,
                  safe_reduce_times_float_mat3x2, safe_reduce_domain_point_times_float_mat3x2,
                  FieldTimesOpFloatMat3x2, float, float_6, int, MUL, MUL, 6, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat3x2,
                  safe_reduce_times_double_mat3x2, safe_reduce_domain_point_times_double_mat3x2,
                  FieldTimesOpDoubleMat3x2, double, double_6, size_t, MUL, MUL, 6, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat3x2,
                  safe_reduce_times_int32_mat3x2, safe_reduce_domain_point_times_int32_mat3x2,
                  FieldTimesOpIntMat3x2, int, int_6, int, MUL, MUL, 6, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat3x3,
                  safe_reduce_times_float_mat3x3, safe_reduce_domain_point_times_float_mat3x3,
                  FieldTimesOpFloatMat3x3, float, float_9, int, MUL, MUL, 9,  0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat3x3,
                  safe_reduce_times_double_mat3x3, safe_reduce_domain_point_times_double_mat3x3,
                  FieldTimesOpDoubleMat3x3, double, double_9, size_t, MUL, MUL, 9, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat3x3,
                  safe_reduce_times_int32_mat3x3, safe_reduce_domain_point_times_int32_mat3x3,
                  FieldTimesOpIntMat3x3, int, int_9, int, MUL, MUL, 9, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat3x4,
                  safe_reduce_times_float_mat3x4, safe_reduce_domain_point_times_float_mat3x4,
                  FieldTimesOpFloatMat3x4, float, float_12, int, MUL, MUL, 12, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat3x4,
                  safe_reduce_times_double_mat3x4, safe_reduce_domain_point_times_double_mat3x4,
                  FieldTimesOpDoubleMat3x4, double, double_12, size_t, MUL, MUL, 12, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat3x4,
                  safe_reduce_times_int32_mat3x4, safe_reduce_domain_point_times_int32_mat3x4,
                  FieldTimesOpIntMat3x4, int, int_12, int, MUL, MUL, 12, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat4x2,
                  safe_reduce_times_float_mat4x2, safe_reduce_domain_point_times_float_mat4x2,
                  FieldTimesOpFloatMat4x2, float, float_8, int, MUL, MUL, 8, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat4x2,
                  safe_reduce_times_double_mat4x2, safe_reduce_domain_point_times_double_mat4x2,
                  FieldTimesOpDoubleMat4x2, double, double_8, size_t, MUL, MUL, 8, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat4x2,
                  safe_reduce_times_int32_mat4x2, safe_reduce_domain_point_times_int32_mat4x2,
                  FieldTimesOpIntMat4x2, int, int_8, int, MUL, MUL, 8, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat4x3,
                  safe_reduce_times_float_mat4x3, safe_reduce_domain_point_times_float_mat4x3,
                  FieldTimesOpFloatMat4x3, float, float_12, int, MUL, MUL, 12,  0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat4x3,
                  safe_reduce_times_double_mat4x3, safe_reduce_domain_point_times_double_mat4x3,
                  FieldTimesOpDoubleMat4x3, double, double_12, size_t, MUL, MUL, 12, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat4x3,
                  safe_reduce_times_int32_mat4x3, safe_reduce_domain_point_times_int32_mat4x3,
                  FieldTimesOpIntMat4x3, int, int_12, int, MUL, MUL, 12, 0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_float_mat4x4,
                  safe_reduce_times_float_mat4x4, safe_reduce_domain_point_times_float_mat4x4,
                  FieldTimesOpFloatMat4x4, float, float_16, int, MUL, MUL, 16, 0.0f)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_double_mat4x4,
                  safe_reduce_times_double_mat4x4, safe_reduce_domain_point_times_double_mat4x4,
                  FieldTimesOpDoubleMat4x4, double, double_16, size_t, MUL, MUL, 16, 0.0)
DECLARE_FIELD_REDUCTION(register_reduction_field_times_int32_mat4x4,
                  safe_reduce_times_int32_mat4x4, safe_reduce_domain_point_times_int32_mat4x4,
                  FieldTimesOpIntMat4x4, int, int_16, int, MUL, MUL, 16, 0)

// declare max reductions on scalars
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float,
                  safe_reduce_max_float, safe_reduce_domain_point_max_float,
                  FieldMaxOpFloat, float, float_1, int, max, max, 1, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double,
                  safe_reduce_max_double, safe_reduce_domain_point_max_double,
                  FieldMaxOpDouble, double, double_1, size_t, max, max, 1, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32,
                  safe_reduce_max_int32, safe_reduce_domain_point_max_int32,
                  FieldMaxOpInt, int, int_1, int, max, max, 1, INT_MAX)

// declare max reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_vec2,
                  safe_reduce_max_float_vec2, safe_reduce_domain_point_max_float_vec2,
                  FieldMaxOpFloatVec2, float, float_2, int, max, max, 2, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_vec2,
                  safe_reduce_max_double_vec2, safe_reduce_domain_point_max_double_vec2,
                  FieldMaxOpDoubleVec2, double, double_2, size_t, max, max, 2, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_vec2,
                  safe_reduce_max_int32_vec2, safe_reduce_domain_point_max_int32_vec2,
                  FieldMaxOpIntVec2, int, int_2, int, max, max, 2, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_vec3,
                  safe_reduce_max_float_vec3, safe_reduce_domain_point_max_float_vec3,
                  FieldMaxOpFloatVec3, float, float_3, int, max, max, 3, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_vec3,
                  safe_reduce_max_double_vec3, safe_reduce_domain_point_max_double_vec3,
                  FieldMaxOpDoubleVec3, double, double_3, size_t, max, max, 3, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_vec3,
                  safe_reduce_max_int32_vec3, safe_reduce_domain_point_max_int32_vec3,
                  FieldMaxOpIntVec3, int, int_3, int, max, max, 3, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_vec4,
                  safe_reduce_max_float_vec4, safe_reduce_domain_point_max_float_vec4,
                  FieldMaxOpFloatVec4, float, float_4, int, max, max, 4, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_vec4,
                  safe_reduce_max_double_vec4, safe_reduce_domain_point_max_double_vec4,
                  FieldMaxOpDoubleVec4, double, double_4, size_t, max, max, 4, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_vec4,
                  safe_reduce_max_int32_vec4, safe_reduce_domain_point_max_int32_vec4,
                  FieldMaxOpIntVec4, int, int_4, int, max, max, 4, INT_MAX)

// declare max reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat2x2,
                  safe_reduce_max_float_mat2x2, safe_reduce_domain_point_max_float_mat2x2,
                  FieldMaxOpFloatMat2x2, float, float_4, int, max, max, 4, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat2x2,
                  safe_reduce_max_double_mat2x2, safe_reduce_domain_point_max_double_mat2x2,
                  FieldMaxOpDoubleMat2x2, double, double_4, size_t, max, max, 4, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat2x2,
                  safe_reduce_max_int32_mat2x2, safe_reduce_domain_point_max_int32_mat2x2,
                  FieldMaxOpIntMat2x2, int, int_4, int, max, max, 4, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat2x3,
                  safe_reduce_max_float_mat2x3, safe_reduce_domain_point_max_float_mat2x3,
                  FieldMaxOpFloatMat2x3, float, float_6, int, max, max, 6,  -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat2x3,
                  safe_reduce_max_double_mat2x3, safe_reduce_domain_point_max_double_mat2x3,
                  FieldMaxOpDoubleMat2x3, double, double_6, size_t, max, max, 6, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat2x3,
                  safe_reduce_max_int32_mat2x3, safe_reduce_domain_point_max_int32_mat2x3,
                  FieldMaxOpIntMat2x3, int, int_6, int, max, max, 6, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat2x4,
                  safe_reduce_max_float_mat2x4, safe_reduce_domain_point_max_float_mat2x4,
                  FieldMaxOpFloatMat2x4, float, float_8, int, max, max, 8, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat2x4,
                  safe_reduce_max_double_mat2x4, safe_reduce_domain_point_max_double_mat2x4,
                  FieldMaxOpDoubleMat2x4, double, double_8, size_t, max, max, 8, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat2x4,
                  safe_reduce_max_int32_mat2x4, safe_reduce_domain_point_max_int32_mat2x4,
                  FieldMaxOpIntMat2x4, int, int_8, int, max, max, 8, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat3x2,
                  safe_reduce_max_float_mat3x2, safe_reduce_domain_point_max_float_mat3x2,
                  FieldMaxOpFloatMat3x2, float, float_6, int, max, max, 6, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat3x2,
                  safe_reduce_max_double_mat3x2, safe_reduce_domain_point_max_double_mat3x2,
                  FieldMaxOpDoubleMat3x2, double, double_6, size_t, max, max, 6, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat3x2,
                  safe_reduce_max_int32_mat3x2, safe_reduce_domain_point_max_int32_mat3x2,
                  FieldMaxOpIntMat3x2, int, int_6, int, max, max, 6, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat3x3,
                  safe_reduce_max_float_mat3x3, safe_reduce_domain_point_max_float_mat3x3,
                  FieldMaxOpFloatMat3x3, float, float_9, int, max, max, 9,  -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat3x3,
                  safe_reduce_max_double_mat3x3, safe_reduce_domain_point_max_double_mat3x3,
                  FieldMaxOpDoubleMat3x3, double, double_9, size_t, max, max, 9, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat3x3,
                  safe_reduce_max_int32_mat3x3, safe_reduce_domain_point_max_int32_mat3x3,
                  FieldMaxOpIntMat3x3, int, int_9, int, max, max, 9, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat3x4,
                  safe_reduce_max_float_mat3x4, safe_reduce_domain_point_max_float_mat3x4,
                  FieldMaxOpFloatMat3x4, float, float_12, int, max, max, 12, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat3x4,
                  safe_reduce_max_double_mat3x4, safe_reduce_domain_point_max_double_mat3x4,
                  FieldMaxOpDoubleMat3x4, double, double_12, size_t, max, max, 12, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat3x4,
                  safe_reduce_max_int32_mat3x4, safe_reduce_domain_point_max_int32_mat3x4,
                  FieldMaxOpIntMat3x4, int, int_12, int, max, max, 12, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat4x2,
                  safe_reduce_max_float_mat4x2, safe_reduce_domain_point_max_float_mat4x2,
                  FieldMaxOpFloatMat4x2, float, float_8, int, max, max, 8, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat4x2,
                  safe_reduce_max_double_mat4x2, safe_reduce_domain_point_max_double_mat4x2,
                  FieldMaxOpDoubleMat4x2, double, double_8, size_t, max, max, 8, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat4x2,
                  safe_reduce_max_int32_mat4x2, safe_reduce_domain_point_max_int32_mat4x2,
                  FieldMaxOpIntMat4x2, int, int_8, int, max, max, 8, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat4x3,
                  safe_reduce_max_float_mat4x3, safe_reduce_domain_point_max_float_mat4x3,
                  FieldMaxOpFloatMat4x3, float, float_12, int, max, max, 12,  -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat4x3,
                  safe_reduce_max_double_mat4x3, safe_reduce_domain_point_max_double_mat4x3,
                  FieldMaxOpDoubleMat4x3, double, double_12, size_t, max, max, 12, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat4x3,
                  safe_reduce_max_int32_mat4x3, safe_reduce_domain_point_max_int32_mat4x3,
                  FieldMaxOpIntMat4x3, int, int_12, int, max, max, 12, INT_MAX)
DECLARE_FIELD_REDUCTION(register_reduction_field_max_float_mat4x4,
                  safe_reduce_max_float_mat4x4, safe_reduce_domain_point_max_float_mat4x4,
                  FieldMaxOpFloatMat4x4, float, float_16, int, max, max, 16, -std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_double_mat4x4,
                  safe_reduce_max_double_mat4x4, safe_reduce_domain_point_max_double_mat4x4,
                  FieldMaxOpDoubleMat4x4, double, double_16, size_t, max, max, 16, -std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_max_int32_mat4x4,
                  safe_reduce_max_int32_mat4x4, safe_reduce_domain_point_max_int32_mat4x4,
                  FieldMaxOpIntMat4x4, int, int_16, int, max, max, 16, INT_MAX)

// declare min reductions on scalars
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float,
                  safe_reduce_min_float, safe_reduce_domain_point_min_float,
                  FieldMinOpFloat, float, float_1, int, min, min, 1, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double,
                  safe_reduce_min_double, safe_reduce_domain_point_min_double,
                  FieldMinOpDouble, double, double_1, size_t, min, min, 1, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32,
                  safe_reduce_min_int32, safe_reduce_domain_point_min_int32,
                  FieldMinOpInt, int, int_1, int, min, min, 1, INT_MIN)

// declare min reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_vec2,
                  safe_reduce_min_float_vec2, safe_reduce_domain_point_min_float_vec2,
                  FieldMinOpFloatVec2, float, float_2, int, min, min, 2, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_vec2,
                  safe_reduce_min_double_vec2, safe_reduce_domain_point_min_double_vec2,
                  FieldMinOpDoubleVec2, double, double_2, size_t, min, min, 2, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_vec2,
                  safe_reduce_min_int32_vec2, safe_reduce_domain_point_min_int32_vec2,
                  FieldMinOpIntVec2, int, int_2, int, min, min, 2, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_vec3,
                  safe_reduce_min_float_vec3, safe_reduce_domain_point_min_float_vec3,
                  FieldMinOpFloatVec3, float, float_3, int, min, min, 3, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_vec3,
                  safe_reduce_min_double_vec3, safe_reduce_domain_point_min_double_vec3,
                  FieldMinOpDoubleVec3, double, double_3, size_t, min, min, 3, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_vec3,
                  safe_reduce_min_int32_vec3, safe_reduce_domain_point_min_int32_vec3,
                  FieldMinOpIntVec3, int, int_3, int, min, min, 3, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_vec4,
                  safe_reduce_min_float_vec4, safe_reduce_domain_point_min_float_vec4,
                  FieldMinOpFloatVec4, float, float_4, int, min, min, 4, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_vec4,
                  safe_reduce_min_double_vec4, safe_reduce_domain_point_min_double_vec4,
                  FieldMinOpDoubleVec4, double, double_4, size_t, min, min, 4, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_vec4,
                  safe_reduce_min_int32_vec4, safe_reduce_domain_point_min_int32_vec4,
                  FieldMinOpIntVec4, int, int_4, int, min, min, 4, INT_MIN)

// declare min reductions on vectors
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat2x2,
                  safe_reduce_min_float_mat2x2, safe_reduce_domain_point_min_float_mat2x2,
                  FieldMinOpFloatMat2x2, float, float_4, int, min, min, 4, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat2x2,
                  safe_reduce_min_double_mat2x2, safe_reduce_domain_point_min_double_mat2x2,
                  FieldMinOpDoubleMat2x2, double, double_4, size_t, min, min, 4, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat2x2,
                  safe_reduce_min_int32_mat2x2, safe_reduce_domain_point_min_int32_mat2x2,
                  FieldMinOpIntMat2x2, int, int_4, int, min, min, 4, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat2x3,
                  safe_reduce_min_float_mat2x3, safe_reduce_domain_point_min_float_mat2x3,
                  FieldMinOpFloatMat2x3, float, float_6, int, min, min, 6,  +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat2x3,
                  safe_reduce_min_double_mat2x3, safe_reduce_domain_point_min_double_mat2x3,
                  FieldMinOpDoubleMat2x3, double, double_6, size_t, min, min, 6, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat2x3,
                  safe_reduce_min_int32_mat2x3, safe_reduce_domain_point_min_int32_mat2x3,
                  FieldMinOpIntMat2x3, int, int_6, int, min, min, 6, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat2x4,
                  safe_reduce_min_float_mat2x4, safe_reduce_domain_point_min_float_mat2x4,
                  FieldMinOpFloatMat2x4, float, float_8, int, min, min, 8, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat2x4,
                  safe_reduce_min_double_mat2x4, safe_reduce_domain_point_min_double_mat2x4,
                  FieldMinOpDoubleMat2x4, double, double_8, size_t, min, min, 8, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat2x4,
                  safe_reduce_min_int32_mat2x4, safe_reduce_domain_point_min_int32_mat2x4,
                  FieldMinOpIntMat2x4, int, int_8, int, min, min, 8, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat3x2,
                  safe_reduce_min_float_mat3x2, safe_reduce_domain_point_min_float_mat3x2,
                  FieldMinOpFloatMat3x2, float, float_6, int, min, min, 6, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat3x2,
                  safe_reduce_min_double_mat3x2, safe_reduce_domain_point_min_double_mat3x2,
                  FieldMinOpDoubleMat3x2, double, double_6, size_t, min, min, 6, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat3x2,
                  safe_reduce_min_int32_mat3x2, safe_reduce_domain_point_min_int32_mat3x2,
                  FieldMinOpIntMat3x2, int, int_6, int, min, min, 6, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat3x3,
                  safe_reduce_min_float_mat3x3, safe_reduce_domain_point_min_float_mat3x3,
                  FieldMinOpFloatMat3x3, float, float_9, int, min, min, 9,  +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat3x3,
                  safe_reduce_min_double_mat3x3, safe_reduce_domain_point_min_double_mat3x3,
                  FieldMinOpDoubleMat3x3, double, double_9, size_t, min, min, 9, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat3x3,
                  safe_reduce_min_int32_mat3x3, safe_reduce_domain_point_min_int32_mat3x3,
                  FieldMinOpIntMat3x3, int, int_9, int, min, min, 9, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat3x4,
                  safe_reduce_min_float_mat3x4, safe_reduce_domain_point_min_float_mat3x4,
                  FieldMinOpFloatMat3x4, float, float_12, int, min, min, 12, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat3x4,
                  safe_reduce_min_double_mat3x4, safe_reduce_domain_point_min_double_mat3x4,
                  FieldMinOpDoubleMat3x4, double, double_12, size_t, min, min, 12, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat3x4,
                  safe_reduce_min_int32_mat3x4, safe_reduce_domain_point_min_int32_mat3x4,
                  FieldMinOpIntMat3x4, int, int_12, int, min, min, 12, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat4x2,
                  safe_reduce_min_float_mat4x2, safe_reduce_domain_point_min_float_mat4x2,
                  FieldMinOpFloatMat4x2, float, float_8, int, min, min, 8, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat4x2,
                  safe_reduce_min_double_mat4x2, safe_reduce_domain_point_min_double_mat4x2,
                  FieldMinOpDoubleMat4x2, double, double_8, size_t, min, min, 8, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat4x2,
                  safe_reduce_min_int32_mat4x2, safe_reduce_domain_point_min_int32_mat4x2,
                  FieldMinOpIntMat4x2, int, int_8, int, min, min, 8, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat4x3,
                  safe_reduce_min_float_mat4x3, safe_reduce_domain_point_min_float_mat4x3,
                  FieldMinOpFloatMat4x3, float, float_12, int, min, min, 12,  +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat4x3,
                  safe_reduce_min_double_mat4x3, safe_reduce_domain_point_min_double_mat4x3,
                  FieldMinOpDoubleMat4x3, double, double_12, size_t, min, min, 12, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat4x3,
                  safe_reduce_min_int32_mat4x3, safe_reduce_domain_point_min_int32_mat4x3,
                  FieldMinOpIntMat4x3, int, int_12, int, min, min, 12, INT_MIN)
DECLARE_FIELD_REDUCTION(register_reduction_field_min_float_mat4x4,
                  safe_reduce_min_float_mat4x4, safe_reduce_domain_point_min_float_mat4x4,
                  FieldMinOpFloatMat4x4, float, float_16, int, min, min, 16, +std::numeric_limits<float>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_double_mat4x4,
                  safe_reduce_min_double_mat4x4, safe_reduce_domain_point_min_double_mat4x4,
                  FieldMinOpDoubleMat4x4, double, double_16, size_t, min, min, 16, +std::numeric_limits<double>::infinity())
DECLARE_FIELD_REDUCTION(register_reduction_field_min_int32_mat4x4,
                  safe_reduce_min_int32_mat4x4, safe_reduce_domain_point_min_int32_mat4x4,
                  FieldMinOpIntMat4x4, int, int_16, int, min, min, 16, INT_MIN)


// GLOBAL (FUTURE) DATA

// Pre-defined reduction operators
#define DECLARE_GLOBAL_REDUCTION(REG, CLASS, T, T_N, U, APPLY_OP, FOLD_OP, N, ID) \
  class CLASS {                                                         \
  public:                                                               \
  typedef TaskResult LHS;                                               \
  typedef TaskResult RHS;                                               \
  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);       \
  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);      \
  static const T_N identity_buffer;                                     \
  static const TaskResult identity;                                     \
  };                                                                    \
                                                                        \
  const T_N CLASS::identity_buffer = { ID };                            \
  const TaskResult CLASS::identity((void *)&CLASS::identity_buffer,     \
                                   sizeof(CLASS::identity_buffer));     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<true>(LHS &lhs_, RHS rhs_)                          \
  {                                                                     \
    assert(lhs_.value_size == sizeof(T_N));                             \
    assert(rhs_.value_size == sizeof(T_N));                             \
    T_N &lhs = *(T_N *)(lhs_.value);                                    \
    T_N &rhs = *(T_N *)(rhs_.value);                                     \
    for (int i = 0; i < N; ++i) {                                       \
      lhs.value[i] = APPLY_OP(lhs.value[i], rhs.value[i]);              \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<false>(LHS &lhs_, RHS rhs_)                         \
  {                                                                     \
    assert(lhs_.value_size == sizeof(T_N));                             \
    assert(rhs_.value_size == sizeof(T_N));                             \
    T_N &lhs = *(T_N *)(lhs_.value);                                    \
    T_N &rhs = *(T_N *)(rhs_.value);                                     \
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
  void CLASS::fold<true>(RHS &rhs1_, RHS rhs2_)                         \
  {                                                                     \
    assert(rhs1_.value_size == sizeof(T_N));                            \
    assert(rhs2_.value_size == sizeof(T_N));                            \
    T_N &rhs1 = *(T_N *)(rhs1_.value);                                  \
    T_N &rhs2 = *(T_N *)(rhs2_.value);                                   \
    for (int i = 0; i < N; ++i) {                                       \
      rhs1.value[i] = FOLD_OP(rhs1.value[i], rhs2.value[i]);            \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<false>(RHS &rhs1_, RHS rhs2_)                        \
  {                                                                     \
    assert(rhs1_.value_size == sizeof(T_N));                            \
    assert(rhs2_.value_size == sizeof(T_N));                            \
    T_N &rhs1 = *(T_N *)(rhs1_.value);                                  \
    T_N &rhs2 = *(T_N *)(rhs2_.value);                                   \
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
  }                                                                     \


// declare plus reductions on scalars
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float,
                  GlobalPlusOpfloat, float, float_1, int, ADD, ADD, 1, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double,
                  GlobalPlusOpdouble, double, double_1, size_t, ADD, ADD, 1, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32,
                  GlobalPlusOpint, int, int_1, int, ADD, ADD, 1, 0)

// declare plus reductions on vectors
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_vec2,
                  GlobalPlusOpred_floatVec2, float, float_2, int, ADD, ADD, 2, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_vec2,
                  GlobalPlusOpred_doubleVec2, double, double_2, size_t, ADD, ADD, 2, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_vec2,
                  GlobalPlusOpred_intVec2, int, int_2, int, ADD, ADD, 2, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_vec3,
                  GlobalPlusOpred_floatVec3, float, float_3, int, ADD, ADD, 3, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_vec3,
                  GlobalPlusOpred_doubleVec3, double, double_3, size_t, ADD, ADD, 3, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_vec3,
                  GlobalPlusOpred_intVec3, int, int_3, int, ADD, ADD, 3, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_vec4,
                  GlobalPlusOpred_floatVec4, float, float_4, int, ADD, ADD, 4, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_vec4,
                  GlobalPlusOpred_doubleVec4, double, double_4, size_t, ADD, ADD, 4, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_vec4,
                  GlobalPlusOpred_intVec4, int, int_4, int, ADD, ADD, 4, 0)

// declare plus reductions on matrices
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat2x2,
                  GlobalPlusOpred_floatMat2x2, float, float_4, int, ADD, ADD, 4, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat2x2,
                  GlobalPlusOpred_doubleMat2x2, double, double_4, size_t, ADD, ADD, 4, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat2x2,
                  GlobalPlusOpred_intMat2x2, int, int_4, int, ADD, ADD, 4, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat2x3,
                  GlobalPlusOpred_floatMat2x3, float, float_6, int, ADD, ADD, 6,  0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat2x3,
                  GlobalPlusOpred_doubleMat2x3, double, double_6, size_t, ADD, ADD, 6, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat2x3,
                  GlobalPlusOpred_intMat2x3, int, int_6, int, ADD, ADD, 6, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat2x4,
                  GlobalPlusOpred_floatMat2x4, float, float_8, int, ADD, ADD, 8, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat2x4,
                  GlobalPlusOpred_doubleMat2x4, double, double_8, size_t, ADD, ADD, 8, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat2x4,
                  GlobalPlusOpred_intMat2x4, int, int_8, int, ADD, ADD, 8, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat3x2,
                  GlobalPlusOpred_floatMat3x2, float, float_6, int, ADD, ADD, 6, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat3x2,
                  GlobalPlusOpred_doubleMat3x2, double, double_6, size_t, ADD, ADD, 6, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat3x2,
                  GlobalPlusOpred_intMat3x2, int, int_6, int, ADD, ADD, 6, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat3x3,
                  GlobalPlusOpred_floatMat3x3, float, float_9, int, ADD, ADD, 9,  0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat3x3,
                  GlobalPlusOpred_doubleMat3x3, double, double_9, size_t, ADD, ADD, 9, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat3x3,
                  GlobalPlusOpred_intMat3x3, int, int_9, int, ADD, ADD, 9, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat3x4,
                  GlobalPlusOpred_floatMat3x4, float, float_12, int, ADD, ADD, 12, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat3x4,
                  GlobalPlusOpred_doubleMat3x4, double, double_12, size_t, ADD, ADD, 12, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat3x4,
                  GlobalPlusOpred_intMat3x4, int, int_12, int, ADD, ADD, 12, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat4x2,
                  GlobalPlusOpred_floatMat4x2, float, float_8, int, ADD, ADD, 8, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat4x2,
                  GlobalPlusOpred_doubleMat4x2, double, double_8, size_t, ADD, ADD, 8, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat4x2,
                  GlobalPlusOpred_intMat4x2, int, int_8, int, ADD, ADD, 8, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat4x3,
                  GlobalPlusOpred_floatMat4x3, float, float_12, int, ADD, ADD, 12,  0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat4x3,
                  GlobalPlusOpred_doubleMat4x3, double, double_12, size_t, ADD, ADD, 12, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat4x3,
                  GlobalPlusOpred_intMat4x3, int, int_12, int, ADD, ADD, 12, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_float_mat4x4,
                  GlobalPlusOpred_floatMat4x4, float, float_16, int, ADD, ADD, 16, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_double_mat4x4,
                  GlobalPlusOpred_doubleMat4x4, double, double_16, size_t, ADD, ADD, 16, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32_mat4x4,
                  GlobalPlusOpred_intMat4x4, int, int_16, int, ADD, ADD, 16, 0)

// declare times reductions on scalars
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float,
                  GlobalTimesOpfloat, float, float_1, int, MUL, MUL, 1, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double,
                  GlobalTimesOpdouble, double, double_1, size_t, MUL, MUL, 1, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32,
                  GlobalTimesOpint, int, int_1, int, MUL, MUL, 1, 0)

// declare times reductions on vectors
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_vec2,
                  GlobalTimesOpred_floatVec2, float, float_2, int, MUL, MUL, 2, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_vec2,
                  GlobalTimesOpred_doubleVec2, double, double_2, size_t, MUL, MUL, 2, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_vec2,
                  GlobalTimesOpred_intVec2, int, int_2, int, MUL, MUL, 2, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_vec3,
                  GlobalTimesOpred_floatVec3, float, float_3, int, MUL, MUL, 3, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_vec3,
                  GlobalTimesOpred_doubleVec3, double, double_3, size_t, MUL, MUL, 3, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_vec3,
                  GlobalTimesOpred_intVec3, int, int_3, int, MUL, MUL, 3, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_vec4,
                  GlobalTimesOpred_floatVec4, float, float_4, int, MUL, MUL, 4, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_vec4,
                  GlobalTimesOpred_doubleVec4, double, double_4, size_t, MUL, MUL, 4, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_vec4,
                  GlobalTimesOpred_intVec4, int, int_4, int, MUL, MUL, 4, 0)

// declare times reductions on matrices
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat2x2,
                  GlobalTimesOpred_floatMat2x2, float, float_4, int, MUL, MUL, 4, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat2x2,
                  GlobalTimesOpred_doubleMat2x2, double, double_4, size_t, MUL, MUL, 4, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat2x2,
                  GlobalTimesOpred_intMat2x2, int, int_4, int, MUL, MUL, 4, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat2x3,
                  GlobalTimesOpred_floatMat2x3, float, float_6, int, MUL, MUL, 6,  0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat2x3,
                  GlobalTimesOpred_doubleMat2x3, double, double_6, size_t, MUL, MUL, 6, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat2x3,
                  GlobalTimesOpred_intMat2x3, int, int_6, int, MUL, MUL, 6, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat2x4,
                  GlobalTimesOpred_floatMat2x4, float, float_8, int, MUL, MUL, 8, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat2x4,
                  GlobalTimesOpred_doubleMat2x4, double, double_8, size_t, MUL, MUL, 8, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat2x4,
                  GlobalTimesOpred_intMat2x4, int, int_8, int, MUL, MUL, 8, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat3x2,
                  GlobalTimesOpred_floatMat3x2, float, float_6, int, MUL, MUL, 6, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat3x2,
                  GlobalTimesOpred_doubleMat3x2, double, double_6, size_t, MUL, MUL, 6, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat3x2,
                  GlobalTimesOpred_intMat3x2, int, int_6, int, MUL, MUL, 6, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat3x3,
                  GlobalTimesOpred_floatMat3x3, float, float_9, int, MUL, MUL, 9,  0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat3x3,
                  GlobalTimesOpred_doubleMat3x3, double, double_9, size_t, MUL, MUL, 9, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat3x3,
                  GlobalTimesOpred_intMat3x3, int, int_9, int, MUL, MUL, 9, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat3x4,
                  GlobalTimesOpred_floatMat3x4, float, float_12, int, MUL, MUL, 12, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat3x4,
                  GlobalTimesOpred_doubleMat3x4, double, double_12, size_t, MUL, MUL, 12, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat3x4,
                  GlobalTimesOpred_intMat3x4, int, int_12, int, MUL, MUL, 12, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat4x2,
                  GlobalTimesOpred_floatMat4x2, float, float_8, int, MUL, MUL, 8, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat4x2,
                  GlobalTimesOpred_doubleMat4x2, double, double_8, size_t, MUL, MUL, 8, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat4x2,
                  GlobalTimesOpred_intMat4x2, int, int_8, int, MUL, MUL, 8, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat4x3,
                  GlobalTimesOpred_floatMat4x3, float, float_12, int, MUL, MUL, 12,  0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat4x3,
                  GlobalTimesOpred_doubleMat4x3, double, double_12, size_t, MUL, MUL, 12, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat4x3,
                  GlobalTimesOpred_intMat4x3, int, int_12, int, MUL, MUL, 12, 0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_float_mat4x4,
                  GlobalTimesOpred_floatMat4x4, float, float_16, int, MUL, MUL, 16, 0.0f)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_double_mat4x4,
                  GlobalTimesOpred_doubleMat4x4, double, double_16, size_t, MUL, MUL, 16, 0.0)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_times_int32_mat4x4,
                  GlobalTimesOpred_intMat4x4, int, int_16, int, MUL, MUL, 16, 0)

// declare max reductions on scalars
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float,
                  GlobalMaxOpfloat, float, float_1, int, max, max, 1, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double,
                  GlobalMaxOpdouble, double, double_1, size_t, max, max, 1, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32,
                  GlobalMaxOpint, int, int_1, int, max, max, 1, INT_MAX)

// declare max reductions on vectors
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_vec2,
                  GlobalMaxOpred_floatVec2, float, float_2, int, max, max, 2, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_vec2,
                  GlobalMaxOpred_doubleVec2, double, double_2, size_t, max, max, 2, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_vec2,
                  GlobalMaxOpred_intVec2, int, int_2, int, max, max, 2, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_vec3,
                  GlobalMaxOpred_floatVec3, float, float_3, int, max, max, 3, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_vec3,
                  GlobalMaxOpred_doubleVec3, double, double_3, size_t, max, max, 3, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_vec3,
                  GlobalMaxOpred_intVec3, int, int_3, int, max, max, 3, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_vec4,
                  GlobalMaxOpred_floatVec4, float, float_4, int, max, max, 4, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_vec4,
                  GlobalMaxOpred_doubleVec4, double, double_4, size_t, max, max, 4, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_vec4,
                  GlobalMaxOpred_intVec4, int, int_4, int, max, max, 4, INT_MAX)

// declare max reductions on matrices
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat2x2,
                  GlobalMaxOpred_floatMat2x2, float, float_4, int, max, max, 4, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat2x2,
                  GlobalMaxOpred_doubleMat2x2, double, double_4, size_t, max, max, 4, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat2x2,
                  GlobalMaxOpred_intMat2x2, int, int_4, int, max, max, 4, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat2x3,
                  GlobalMaxOpred_floatMat2x3, float, float_6, int, max, max, 6,  -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat2x3,
                  GlobalMaxOpred_doubleMat2x3, double, double_6, size_t, max, max, 6, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat2x3,
                  GlobalMaxOpred_intMat2x3, int, int_6, int, max, max, 6, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat2x4,
                  GlobalMaxOpred_floatMat2x4, float, float_8, int, max, max, 8, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat2x4,
                  GlobalMaxOpred_doubleMat2x4, double, double_8, size_t, max, max, 8, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat2x4,
                  GlobalMaxOpred_intMat2x4, int, int_8, int, max, max, 8, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat3x2,
                  GlobalMaxOpred_floatMat3x2, float, float_6, int, max, max, 6, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat3x2,
                  GlobalMaxOpred_doubleMat3x2, double, double_6, size_t, max, max, 6, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat3x2,
                  GlobalMaxOpred_intMat3x2, int, int_6, int, max, max, 6, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat3x3,
                  GlobalMaxOpred_floatMat3x3, float, float_9, int, max, max, 9,  -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat3x3,
                  GlobalMaxOpred_doubleMat3x3, double, double_9, size_t, max, max, 9, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat3x3,
                  GlobalMaxOpred_intMat3x3, int, int_9, int, max, max, 9, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat3x4,
                  GlobalMaxOpred_floatMat3x4, float, float_12, int, max, max, 12, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat3x4,
                  GlobalMaxOpred_doubleMat3x4, double, double_12, size_t, max, max, 12, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat3x4,
                  GlobalMaxOpred_intMat3x4, int, int_12, int, max, max, 12, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat4x2,
                  GlobalMaxOpred_floatMat4x2, float, float_8, int, max, max, 8, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat4x2,
                  GlobalMaxOpred_doubleMat4x2, double, double_8, size_t, max, max, 8, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat4x2,
                  GlobalMaxOpred_intMat4x2, int, int_8, int, max, max, 8, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat4x3,
                  GlobalMaxOpred_floatMat4x3, float, float_12, int, max, max, 12,  -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat4x3,
                  GlobalMaxOpred_doubleMat4x3, double, double_12, size_t, max, max, 12, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat4x3,
                  GlobalMaxOpred_intMat4x3, int, int_12, int, max, max, 12, INT_MAX)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_float_mat4x4,
                  GlobalMaxOpred_floatMat4x4, float, float_16, int, max, max, 16, -std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_double_mat4x4,
                  GlobalMaxOpred_doubleMat4x4, double, double_16, size_t, max, max, 16, -std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_max_int32_mat4x4,
                  GlobalMaxOpred_intMat4x4, int, int_16, int, max, max, 16, INT_MAX)

// declare min reductions on scalars
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float,
                  GlobalMinOpfloat, float, float_1, int, min, min, 1, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double,
                  GlobalMinOpdouble, double, double_1, size_t, min, min, 1, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32,
                  GlobalMinOpint, int, int_1, int, min, min, 1, INT_MIN)

// declare min reductions on vectors
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_vec2,
                  GlobalMinOpred_floatVec2, float, float_2, int, min, min, 2, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_vec2,
                  GlobalMinOpred_doubleVec2, double, double_2, size_t, min, min, 2, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_vec2,
                  GlobalMinOpred_intVec2, int, int_2, int, min, min, 2, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_vec3,
                  GlobalMinOpred_floatVec3, float, float_3, int, min, min, 3, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_vec3,
                  GlobalMinOpred_doubleVec3, double, double_3, size_t, min, min, 3, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_vec3,
                  GlobalMinOpred_intVec3, int, int_3, int, min, min, 3, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_vec4,
                  GlobalMinOpred_floatVec4, float, float_4, int, min, min, 4, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_vec4,
                  GlobalMinOpred_doubleVec4, double, double_4, size_t, min, min, 4, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_vec4,
                  GlobalMinOpred_intVec4, int, int_4, int, min, min, 4, INT_MIN)

// declare min reductions on matrices
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat2x2,
                  GlobalMinOpred_floatMat2x2, float, float_4, int, min, min, 4, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat2x2,
                  GlobalMinOpred_doubleMat2x2, double, double_4, size_t, min, min, 4, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat2x2,
                  GlobalMinOpred_intMat2x2, int, int_4, int, min, min, 4, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat2x3,
                  GlobalMinOpred_floatMat2x3, float, float_6, int, min, min, 6,  +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat2x3,
                  GlobalMinOpred_doubleMat2x3, double, double_6, size_t, min, min, 6, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat2x3,
                  GlobalMinOpred_intMat2x3, int, int_6, int, min, min, 6, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat2x4,
                  GlobalMinOpred_floatMat2x4, float, float_8, int, min, min, 8, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat2x4,
                  GlobalMinOpred_doubleMat2x4, double, double_8, size_t, min, min, 8, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat2x4,
                  GlobalMinOpred_intMat2x4, int, int_8, int, min, min, 8, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat3x2,
                  GlobalMinOpred_floatMat3x2, float, float_6, int, min, min, 6, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat3x2,
                  GlobalMinOpred_doubleMat3x2, double, double_6, size_t, min, min, 6, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat3x2,
                  GlobalMinOpred_intMat3x2, int, int_6, int, min, min, 6, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat3x3,
                  GlobalMinOpred_floatMat3x3, float, float_9, int, min, min, 9,  +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat3x3,
                  GlobalMinOpred_doubleMat3x3, double, double_9, size_t, min, min, 9, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat3x3,
                  GlobalMinOpred_intMat3x3, int, int_9, int, min, min, 9, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat3x4,
                  GlobalMinOpred_floatMat3x4, float, float_12, int, min, min, 12, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat3x4,
                  GlobalMinOpred_doubleMat3x4, double, double_12, size_t, min, min, 12, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat3x4,
                  GlobalMinOpred_intMat3x4, int, int_12, int, min, min, 12, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat4x2,
                  GlobalMinOpred_floatMat4x2, float, float_8, int, min, min, 8, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat4x2,
                  GlobalMinOpred_doubleMat4x2, double, double_8, size_t, min, min, 8, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat4x2,
                  GlobalMinOpred_intMat4x2, int, int_8, int, min, min, 8, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat4x3,
                  GlobalMinOpred_floatMat4x3, float, float_12, int, min, min, 12,  +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat4x3,
                  GlobalMinOpred_doubleMat4x3, double, double_12, size_t, min, min, 12, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat4x3,
                  GlobalMinOpred_intMat4x3, int, int_12, int, min, min, 12, INT_MIN)
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_float_mat4x4,
                  GlobalMinOpred_floatMat4x4, float, float_16, int, min, min, 16, +std::numeric_limits<float>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_double_mat4x4,
                  GlobalMinOpred_doubleMat4x4, double, double_16, size_t, min, min, 16, +std::numeric_limits<double>::infinity())
DECLARE_GLOBAL_REDUCTION(register_reduction_global_min_int32_mat4x4,
                  GlobalMinOpred_intMat4x4, int, int_16, int, min, min, 16, INT_MIN)
