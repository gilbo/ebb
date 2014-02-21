if not terralib.cudacompile then
	print("This simulation requires CUDA support; exiting...")
	return
end


--------------------------------------------------------------------------------
--[[ Grab references to CUDA API                                            ]]--
--------------------------------------------------------------------------------
terralib.includepath = terralib.includepath..";/usr/local/cuda/include"
local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]

-- Find cudasMemcpyKind enum in <cuda-include>/driver_types.h, have to
-- declare manually since terra doesnt grok enums from include files
C.cudaMemcpyHostToDevice = 1
C.cudaMemcpyDeviceToHost = 2

local tid   = cudalib.nvvm_read_ptx_sreg_tid_x   -- threadId.x
local ntid  = cudalib.nvvm_read_ptx_sreg_ntid_x  -- terralib.intrinsic("llvm.nvvm.read.ptx.sreg.ntid.x",{} -> int)
local sqrt  = cudalib.nvvm_sqrt_rm_d             -- floating point sqrt, round to nearest
local aadd  = terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32", {&float,float} -> {float})


--------------------------------------------------------------------------------
--[[ Read in mesh relation, initialize fields                               ]]--
--------------------------------------------------------------------------------
local PN    = terralib.require 'compiler.pathname'
local LMesh = terralib.require "compiler.lmesh"
local M     = LMesh.Load(PN.scriptdir():concat("rmesh.lmesh"):tostring())

local init_temp    = terra (mem : &float, i : int)
	if i == 0 then
		mem[0] = 1000
	else 
		mem[0] = 0
	end
end

M.vertices:NewField('flux',        L.float):LoadConstant(0)
M.vertices:NewField('jacobistep',  L.float):LoadConstant(0)
M.vertices:NewField('temperature', L.float):LoadFromCallback(init_temp)


--------------------------------------------------------------------------------
--[[ Parallel kernels for GPU:                                              ]]--
--------------------------------------------------------------------------------
local terra compute_step (head : &uint64, tail       : &uint64, 
	                      flux : &float,  jacobistep : &float,
	                      temp : &float,  position   : &float) : {}

	var edge_id : uint64 = tid()
	var head_id : uint64 = head[edge_id]
	var tail_id : uint64 = tail[edge_id]

	var dp : double[3]
	dp[0] = position[3*head_id]   - position[3*tail_id]
	dp[1] = position[3*head_id+1] - position[3*tail_id+1]
	dp[2] = position[3*head_id+2] - position[3*tail_id+2]

	var dpsq = dp[0]*dp[0] + dp[1]*dp[1] + dp[2] * dp[2]
	var len  = sqrt(dpsq)
	var step = 1.0 / len

	var dt : float = temp[head_id] - temp[tail_id]

	-- Atomic floating point add reductions ensure that we don't compute incorrect
	-- results due to race conditions
	aadd(&flux[head_id], -dt * step)
	aadd(&flux[tail_id],  dt * step)

	aadd(&jacobistep[head_id], step)
	aadd(&jacobistep[tail_id], step)
end

local terra propagate_temp (temp : &float, flux : &float, jacobistep : &float)
	var vid = tid()
	temp[vid] = temp[vid] + .01 * flux[vid] / jacobistep[vid]
end

local terra clear_temp_vars (flux : &float, jacobistep : &float)
	var vid = tid()
	flux[vid]       = 0.0
	jacobistep[vid] = 0.0
end

local R = terralib.cudacompile { compute_step    = compute_step,
                                 propagate_temp  = propagate_temp,
                                 clear_temp_vars = clear_temp_vars }


--------------------------------------------------------------------------------
--[[ Simulation:                                                            ]]-- 
--------------------------------------------------------------------------------
terra copy_posn_data (data : &vector(double, 3), N : int) : &float
	var ret : &float = [&float](C.malloc(sizeof(float) * N * 3))
	for i = 0, N do
		ret[3*i]   = data[i][0]
		ret[3*i+1] = data[i][1]
		ret[3*i+2] = data[i][2]
	end
	return ret
end	

local nEdges = M.edges:Size()
local nVerts = M.vertices:Size()

local terra run_simulation (iters : uint64)
	var posn_data = copy_posn_data(M.vertices.position.data, nVerts)

	-- Allocate and copy over field data to GPU device
	var head_ddata : &uint64,
	    tail_ddata : &uint64,
	    flux_ddata : &float,
	    jaco_ddata : &float, 
	    temp_ddata : &float, 
	    posn_ddata : &float

	var tsize = sizeof(uint64) * nEdges -- size of edge topology relation
	var fsize = sizeof(float)  * nVerts -- size of fields over vertices

	C.cudaMalloc([&&opaque](&head_ddata),   tsize)
	C.cudaMalloc([&&opaque](&tail_ddata),   tsize)
	C.cudaMalloc([&&opaque](&flux_ddata),   fsize)
	C.cudaMalloc([&&opaque](&jaco_ddata),   fsize)
	C.cudaMalloc([&&opaque](&temp_ddata),   fsize)
	C.cudaMalloc([&&opaque](&posn_ddata), 3*fsize)

	C.cudaMemcpy([&opaque](head_ddata), [&opaque](M.edges.head.data),             tsize, C.cudaMemcpyHostToDevice)
	C.cudaMemcpy([&opaque](tail_ddata), [&opaque](M.edges.tail.data),             tsize, C.cudaMemcpyHostToDevice)
	C.cudaMemcpy([&opaque](flux_ddata), [&opaque](M.vertices.flux.data),          fsize, C.cudaMemcpyHostToDevice)
	C.cudaMemcpy([&opaque](jaco_ddata), [&opaque](M.vertices.jacobistep.data),    fsize, C.cudaMemcpyHostToDevice)
	C.cudaMemcpy([&opaque](temp_ddata), [&opaque](M.vertices.temperature.data),   fsize, C.cudaMemcpyHostToDevice)
	C.cudaMemcpy([&opaque](posn_ddata), [&opaque](posn_data),                   3*fsize, C.cudaMemcpyHostToDevice)

	-- Launch parameters
	var eLaunch = terralib.CUDAParams { 1, 1, 1, nEdges, 1, 1, 0, nil }
	var vLaunch = terralib.CUDAParams { 1, 1, 1, nVerts, 1, 1, 0, nil }

	-- run kernels!
	for i = 0, iters do
		R.compute_step(&eLaunch,    head_ddata, tail_ddata, flux_ddata,
		                            jaco_ddata, temp_ddata, posn_ddata)
		R.propagate_temp(&vLaunch,  temp_ddata, flux_ddata, jaco_ddata)
		R.clear_temp_vars(&vLaunch, flux_ddata, jaco_ddata)
	end

	-- copy back results
	C.cudaMemcpy([&opaque](M.vertices.temperature.data),  [&opaque](temp_ddata),  fsize, C.cudaMemcpyDeviceToHost)

	-- Free unused memory
	C.free(posn_data)

	C.cudaFree(head_ddata)
	C.cudaFree(tail_ddata)
	C.cudaFree(flux_ddata)
	C.cudaFree(jaco_ddata)
	C.cudaFree(temp_ddata)
	C.cudaFree(posn_ddata)
end

local function main()
	run_simulation(1000)
	-- Debug output:
	M.vertices.temperature:print()
end

main()
