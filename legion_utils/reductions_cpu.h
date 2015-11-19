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

#ifndef __REDUCTIONS_CPU_H__
#define __REDUCTIONS_CPU_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "legion_c.h"

// register plus on scalars
void register_reduction_plus_float(legion_reduction_op_id_t redop);
void register_reduction_plus_double(legion_reduction_op_id_t redop);
void register_reduction_plus_int32(legion_reduction_op_id_t redop);

// register plus on vectors
void register_reduction_plus_float_vec2(legion_reduction_op_id_t redop);
void register_reduction_plus_double_vec2(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_vec2(legion_reduction_op_id_t redop);
void register_reduction_plus_float_vec3(legion_reduction_op_id_t redop);
void register_reduction_plus_double_vec3(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_vec3(legion_reduction_op_id_t redop);
void register_reduction_plus_float_vec4(legion_reduction_op_id_t redop);
void register_reduction_plus_double_vec4(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_vec4(legion_reduction_op_id_t redop);

// register plus on matrices
void register_reduction_plus_float_mat2x2(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat2x2(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat2x2(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat2x3(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat2x3(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat2x3(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat2x4(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat2x4(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat2x4(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat3x2(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat3x2(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat3x2(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat3x3(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat3x3(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat3x3(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat3x4(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat3x4(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat3x4(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat4x2(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat4x2(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat4x2(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat4x3(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat4x3(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat4x3(legion_reduction_op_id_t redop);
void register_reduction_plus_float_mat4x4(legion_reduction_op_id_t redop);
void register_reduction_plus_double_mat4x4(legion_reduction_op_id_t redop);
void register_reduction_plus_int32_mat4x4(legion_reduction_op_id_t redop);

// register times on scalars
void register_reduction_times_float(legion_reduction_op_id_t redop);
void register_reduction_times_double(legion_reduction_op_id_t redop);
void register_reduction_times_int32(legion_reduction_op_id_t redop);

// register times on vectors
void register_reduction_times_float_vec2(legion_reduction_op_id_t redop);
void register_reduction_times_double_vec2(legion_reduction_op_id_t redop);
void register_reduction_times_int32_vec2(legion_reduction_op_id_t redop);
void register_reduction_times_float_vec3(legion_reduction_op_id_t redop);
void register_reduction_times_double_vec3(legion_reduction_op_id_t redop);
void register_reduction_times_int32_vec3(legion_reduction_op_id_t redop);
void register_reduction_times_float_vec4(legion_reduction_op_id_t redop);
void register_reduction_times_double_vec4(legion_reduction_op_id_t redop);
void register_reduction_times_int32_vec4(legion_reduction_op_id_t redop);

// register times on matrices
void register_reduction_times_float_mat2x2(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat2x2(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat2x2(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat2x3(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat2x3(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat2x3(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat2x4(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat2x4(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat2x4(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat3x2(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat3x2(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat3x2(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat3x3(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat3x3(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat3x3(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat3x4(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat3x4(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat3x4(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat4x2(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat4x2(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat4x2(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat4x3(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat4x3(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat4x3(legion_reduction_op_id_t redop);
void register_reduction_times_float_mat4x4(legion_reduction_op_id_t redop);
void register_reduction_times_double_mat4x4(legion_reduction_op_id_t redop);
void register_reduction_times_int32_mat4x4(legion_reduction_op_id_t redop);

// register max on scalars
void register_reduction_max_float(legion_reduction_op_id_t redop);
void register_reduction_max_double(legion_reduction_op_id_t redop);
void register_reduction_max_int32(legion_reduction_op_id_t redop);

// register min on scalars
void register_reduction_min_float(legion_reduction_op_id_t redop);
void register_reduction_min_double(legion_reduction_op_id_t redop);
void register_reduction_min_int32(legion_reduction_op_id_t redop);

// struct for types
typedef struct { float  value[1 ]; }  float_1 ;
typedef struct { double value[1 ]; } double_1 ;
typedef struct { int    value[1 ]; }    int_1 ;
typedef struct { float  value[2 ]; }  float_2 ;
typedef struct { double value[2 ]; } double_2 ;
typedef struct { int    value[2 ]; }    int_2 ;
typedef struct { float  value[3 ]; }  float_3 ;
typedef struct { double value[3 ]; } double_3 ;
typedef struct { int    value[3 ]; }    int_3 ;
typedef struct { float  value[4 ]; }  float_4 ;
typedef struct { double value[4 ]; } double_4 ;
typedef struct { int    value[4 ]; }    int_4 ;
typedef struct { float  value[6 ]; }  float_6 ;
typedef struct { double value[6 ]; } double_6 ;
typedef struct { int    value[6 ]; }    int_6 ;
typedef struct { float  value[8 ]; }  float_8 ;
typedef struct { double value[8 ]; } double_8 ;
typedef struct { int    value[8 ]; }    int_8 ;
typedef struct { float  value[9 ]; }  float_9 ;
typedef struct { double value[9 ]; } double_9 ;
typedef struct { int    value[9 ]; }    int_9 ;
typedef struct { float  value[12]; }  float_12;
typedef struct { double value[12]; } double_12;
typedef struct { int    value[12]; }    int_12;
typedef struct { float  value[16]; }  float_16;
typedef struct { double value[16]; } double_16;
typedef struct { int    value[16]; }    int_16;

// safe reduce plus on scalars
void safe_reduce_plus_float(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, float_1 value);
void safe_reduce_plus_double(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, double_1 value);
void safe_reduce_plus_int32(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, int_1 value);

// safe reduce plus on vectors
void safe_reduce_plus_float_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_2 value);
void safe_reduce_plus_double_vec2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_2 value);
void safe_reduce_plus_int32_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_2 value);
void safe_reduce_plus_float_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_3 value);
void safe_reduce_plus_double_vec3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_3 value);
void safe_reduce_plus_int32_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_3 value);
void safe_reduce_plus_float_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_plus_double_vec4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_plus_int32_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);

// safe reduce plus on matrices
void safe_reduce_plus_float_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_plus_double_mat2x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_plus_int32_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);
void safe_reduce_plus_float_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_plus_double_mat2x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_plus_int32_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_plus_float_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_plus_double_mat2x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_plus_int32_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_plus_float_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_plus_double_mat3x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_plus_int32_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_plus_float_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_9 value);
void safe_reduce_plus_double_mat3x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_9 value);
void safe_reduce_plus_int32_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_9 value);
void safe_reduce_plus_float_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_plus_double_mat3x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_plus_int32_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_plus_float_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_plus_double_mat4x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_plus_int32_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_plus_float_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_plus_double_mat4x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_plus_int32_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_plus_float_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_16 value);
void safe_reduce_plus_double_mat4x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_16 value);
void safe_reduce_plus_int32_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_16 value);

// safe reduce times on scalars
void safe_reduce_times_float(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, float_1 value);
void safe_reduce_times_double(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, double_1 value);
void safe_reduce_times_int32(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, int_1 value);

// safe reduce times on vectors
void safe_reduce_times_float_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_2 value);
void safe_reduce_times_double_vec2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_2 value);
void safe_reduce_times_int32_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_2 value);
void safe_reduce_times_float_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_3 value);
void safe_reduce_times_double_vec3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_3 value);
void safe_reduce_times_int32_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_3 value);
void safe_reduce_times_float_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_times_double_vec4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_times_int32_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);

// safe reduce times on vectors
void safe_reduce_times_float_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_times_double_mat2x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_times_int32_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);
void safe_reduce_times_float_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_times_double_mat2x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_times_int32_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_times_float_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_times_double_mat2x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_times_int32_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_times_float_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_times_double_mat3x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_times_int32_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_times_float_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_9 value);
void safe_reduce_times_double_mat3x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_9 value);
void safe_reduce_times_int32_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_9 value);
void safe_reduce_times_float_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_times_double_mat3x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_times_int32_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_times_float_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_times_double_mat4x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_times_int32_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_times_float_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_times_double_mat4x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_times_int32_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_times_float_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_16 value);
void safe_reduce_times_double_mat4x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_16 value);
void safe_reduce_times_int32_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_16 value);

// safe reduce max on scalars
void safe_reduce_max_float(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, float_1 value);
void safe_reduce_max_double(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, double_1 value);
void safe_reduce_max_int32(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, int_1 value);

// safe reduce max on vectors
void safe_reduce_max_float_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_2 value);
void safe_reduce_max_double_vec2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_2 value);
void safe_reduce_max_int32_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_2 value);
void safe_reduce_max_float_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_3 value);
void safe_reduce_max_double_vec3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_3 value);
void safe_reduce_max_int32_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_3 value);
void safe_reduce_max_float_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_max_double_vec4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_max_int32_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);

// safe reduce max on vectors
void safe_reduce_max_float_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_max_double_mat2x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_max_int32_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);
void safe_reduce_max_float_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_max_double_mat2x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_max_int32_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_max_float_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_max_double_mat2x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_max_int32_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_max_float_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_max_double_mat3x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_max_int32_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_max_float_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_9 value);
void safe_reduce_max_double_mat3x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_9 value);
void safe_reduce_max_int32_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_9 value);
void safe_reduce_max_float_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_max_double_mat3x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_max_int32_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_max_float_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_max_double_mat4x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_max_int32_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_max_float_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_max_double_mat4x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_max_int32_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_max_float_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_16 value);
void safe_reduce_max_double_mat4x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_16 value);
void safe_reduce_max_int32_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_16 value);

// safe reduce min on scalars
void safe_reduce_min_float(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, float_1 value);
void safe_reduce_min_double(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, double_1 value);
void safe_reduce_min_int32(legion_accessor_generic_t accessor,
                            legion_ptr_t ptr, int_1 value);

// safe reduce min on vectors
void safe_reduce_min_float_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_2 value);
void safe_reduce_min_double_vec2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_2 value);
void safe_reduce_min_int32_vec2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_2 value);
void safe_reduce_min_float_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_3 value);
void safe_reduce_min_double_vec3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_3 value);
void safe_reduce_min_int32_vec3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_3 value);
void safe_reduce_min_float_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_min_double_vec4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_min_int32_vec4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);

// safe reduce min on vectors
void safe_reduce_min_float_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_4 value);
void safe_reduce_min_double_mat2x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_4 value);
void safe_reduce_min_int32_mat2x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_4 value);
void safe_reduce_min_float_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_min_double_mat2x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_min_int32_mat2x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_min_float_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_min_double_mat2x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_min_int32_mat2x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_min_float_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_6 value);
void safe_reduce_min_double_mat3x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_6 value);
void safe_reduce_min_int32_mat3x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_6 value);
void safe_reduce_min_float_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_9 value);
void safe_reduce_min_double_mat3x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_9 value);
void safe_reduce_min_int32_mat3x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_9 value);
void safe_reduce_min_float_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_min_double_mat3x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_min_int32_mat3x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_min_float_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_8 value);
void safe_reduce_min_double_mat4x2(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_8 value);
void safe_reduce_min_int32_mat4x2(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_8 value);
void safe_reduce_min_float_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_12 value);
void safe_reduce_min_double_mat4x3(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_12 value);
void safe_reduce_min_int32_mat4x3(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_12 value);
void safe_reduce_min_float_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, float_16 value);
void safe_reduce_min_double_mat4x4(legion_accessor_generic_t accessor,
                                legion_ptr_t ptr, double_16 value);
void safe_reduce_min_int32_mat4x4(legion_accessor_generic_t accessor,
                                 legion_ptr_t ptr, int_16 value);

#ifdef __cplusplus
}
#endif

#endif // __REDUCTIONS_CPU_H__
