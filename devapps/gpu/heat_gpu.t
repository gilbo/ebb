if not terralib.cudacompile then
  print("This simulation requires CUDA support; exiting...")
  return
end
import "ebb"
local L = require "ebblib"

L.SetDefaultProcessor(L.GPU)
--------------------------------------------------------------------------------
--[[ Grab references to CUDA API                                            ]]--
--------------------------------------------------------------------------------
-- CUDA headers are included in compiler.C
-- Enums should be declared there
local C = require "compiler.c"

local tid   = cudalib.nvvm_read_ptx_sreg_tid_x   -- threadId.x
local ntid  = cudalib.nvvm_read_ptx_sreg_ntid_x  -- terralib.intrinsic("llvm.nvvm.read.ptx.sreg.ntid.x",{} -> int)
local sqrt  = cudalib.nvvm_sqrt_rm_d             -- floating point sqrt, round to nearest
local aadd  = terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32", {&float,float} -> {float})


--------------------------------------------------------------------------------
--[[ Read in mesh relation, initialize fields                               ]]--
--------------------------------------------------------------------------------
local PN    = require 'ebb.lib.pathname'
local LMesh = require "ebb.domains.lmesh"
local M     = LMesh.Load(PN.scriptdir():concat("rmesh.lmesh"):tostring())

local function init_temp (i)
  if i == 0 then
    return 1000
  else 
    return 0
  end
end

M.vertices:NewField('flux',        L.float):Load(0)
M.vertices:NewField('jacobistep',  L.float):Load(0)
M.vertices:NewField('temperature', L.float):Load(init_temp)


--------------------------------------------------------------------------------
--[[ Parallel kernels for GPU:                                              ]]--
--------------------------------------------------------------------------------
local terra compute_step (head : &uint64, tail       : &uint64, 
                          flux : &float,  jacobistep : &float,
                          temp : &float,  position   : &double) : {}

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

--local R = terralib.cudacompile { compute_step    = compute_step,
--                                 propagate_temp  = propagate_temp,
--                                 clear_temp_vars = clear_temp_vars }
local R1 = terralib.cudacompile { compute_step = compute_step }
local R2 = terralib.cudacompile { propagate_temp = propagate_temp }
local R3 = terralib.cudacompile { clear_temp_vars = clear_temp_vars }


--------------------------------------------------------------------------------
--[[ Simulation:                                                            ]]-- 
--------------------------------------------------------------------------------
local vec3dtype = M.vertices.position.type:terratype()
terra copy_posn_data (data : &vec3dtype, N : int) : &double
  var ret : &double = [&double](C.malloc(sizeof(double) * N * 3))
  for i = 0, N do
    ret[3*i]   = data[i].d[0]
    ret[3*i+1] = data[i].d[1]
    ret[3*i+2] = data[i].d[2]
  end
  return ret
end 

local nEdges = M.edges:Size()
local nVerts = M.vertices:Size()


local head_ddata = M.edges.head:DataPtr()
local tail_ddata = M.edges.tail:DataPtr()
local flux_ddata = M.vertices.flux:DataPtr()
local jaco_ddata = M.vertices.jacobistep:DataPtr()
local temp_ddata = M.vertices.temperature:DataPtr()
local posn_ddata = M.vertices.position:DataPtr()

local terra run_simulation (iters : uint64)
  -- Launch parameters
  var eLaunch = terralib.CUDAParams { 1, 1, 1, nEdges, 1, 1, 0, nil }
  var vLaunch = terralib.CUDAParams { 1, 1, 1, nVerts, 1, 1, 0, nil }

  -- run kernels!
  for i = 0, iters do
    R1.compute_step(&eLaunch,    head_ddata, tail_ddata, flux_ddata,
                                jaco_ddata, temp_ddata,
                                [&double](posn_ddata))
    R2.propagate_temp(&vLaunch,  temp_ddata, flux_ddata, jaco_ddata)
    R3.clear_temp_vars(&vLaunch, flux_ddata, jaco_ddata)
  end
end

local function main()
  run_simulation(1000)
  -- Debug output:
  M.vertices.temperature:Print()
end

main()
