if not terralib.cudacompile then
  print("This simulation requires CUDA support; exiting...")
  return
end
--[[
This example does a multi-staged reduction of an array on a GPU, by
continually:
 - copying a portion of the global data array to a block and reducing all
   elements on the block, then
 - copying the block result back to the global array.

This example works for array sizes that are a multiple of the block size.
]]


-------------------------------------------------------------------------------
-- Imports                                                                   -- 
-------------------------------------------------------------------------------
import "compiler.liszt"

local tid  = cudalib.nvvm_read_ptx_sreg_tid_x
local bid  = cudalib.nvvm_read_ptx_sreg_ctaid_x

local C = L.require 'compiler.c'


-------------------------------------------------------------------------------
-- Global Parameters                                                         -- 
-------------------------------------------------------------------------------
local threads_per_block = 64
local num_blocks = threads_per_block * threads_per_block * threads_per_block
local N = num_blocks * threads_per_block

-- This symbol can be used to add shared block memory to
-- a terra function intended to run on the GPU.
local treedata = cudalib.sharedmemory(float, threads_per_block)


-------------------------------------------------------------------------------
-- GPU Functions/Kernels                                                     -- 
-------------------------------------------------------------------------------
-- generates an unrolled loop of cascading reductions in shared memory. See
-- NVIDIA white paper on reductions for an idea of what this is trying to do:
-- http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
local function unroll_reduce (ptr, tid)
    local expr = quote end
    local step = threads_per_block

    while (step > 1) do
        step = step/2
        expr = quote
            [expr]
            if tid < step then [ptr][tid] = [ptr][tid] + [ptr][tid + step] end
        end

        -- pairwise reductions over > 32 threads need to be synchronized b/c
        -- they aren't guaranteed to be executed in lockstep, as they are
        -- running in multiple warps
        if step > 32 then
            expr = quote [expr] cudalib.ptx_bar_sync(0) end
        end
    end
    return expr
end

local terra initialize_array(out : &float)
    var t = tid()
    var b = bid()
    var v = t + b * threads_per_block -- global thread id
    out[v] = v
end

local terra staged_global_reduce(input : &float)
    var t = tid()
    var b = bid()
    var gt = t + b * threads_per_block -- global thread id

    -- copy global back into shared
    treedata[t] = input[gt]

    -- reduce chunk in shared memory to one value
    [unroll_reduce(treedata, t)]

    -- copy final value back to global memory
    -- Q: bad to let every thread run this?
    input[b] = treedata[0]
end

local R = terralib.cudacompile({initialize_array     = initialize_array,
                                staged_global_reduce = staged_global_reduce})


-------------------------------------------------------------------------------
-- CPU Functions                                                             -- 
-------------------------------------------------------------------------------
terra launch_single_reduction()
    -- allocate device data
    var data : &float
    C.cudaMalloc([&&opaque](&data),sizeof(float)*N)

    var launch = terralib.CUDAParams { num_blocks,        1, 1,
                                       threads_per_block, 1, 1,
                                       0, nil}

    R.initialize_array(&launch, data)
    R.staged_global_reduce(&launch, data)

    -- copy back partially reduced results
    var result : &float = [&float](C.malloc(sizeof(float)*num_blocks))
    C.cudaMemcpy(result,data,sizeof(float)*num_blocks,2)

    -- Finish reduction on CPU
    var sum : float = 0.0
    for i = 0, num_blocks do
        sum = sum + result[i]
    end
    return sum
end

terra launch_staged_reduction()
    -- allocate device data
	var data : &float
	C.cudaMalloc([&&opaque](&data),sizeof(float)*num_blocks*threads_per_block)

	var launch1 = terralib.CUDAParams { num_blocks,        1, 1,
                                        threads_per_block, 1, 1,
                                        0, nil }
    R.initialize_array(&launch1, data)

    var blocks : uint = num_blocks
    while blocks >= 1 do
        var launch = terralib.CUDAParams { blocks,            1, 1,
                                           threads_per_block, 1, 1,
                                           0, nil }
        R.staged_global_reduce(&launch, data)
        blocks = blocks / threads_per_block
    end

	var result : float
	C.cudaMemcpy(&result,data,sizeof(float),2)
    return result
end


-------------------------------------------------------------------------------
-- Run script:                                                               -- 
-------------------------------------------------------------------------------
local result

if arg[2] == '--single' then
    result = launch_single_reduction()
else
    result = launch_staged_reduction()
end

print('Result: ' .. tostring(result))
