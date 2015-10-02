if not terralib.cudacompile then
  print("This simulation requires CUDA support; exiting...")
  return
end
--[[
This files contains examples for two strategies of array reductions on gpus:
a multi-staged reduction that occurs entirely on the GPU, and a two-stage
reduction that reduces once on the GPU, then copies partially-reduced data
back to the CPU to finish reducing there, and a two-stage reduction that
operates entirely on the GPU.

The multi-staged version works by continually:
 - copying a portion of the global data array to a block and reducing all
   elements on the block, then
 - copying the block result back to the global array.

The GPU/CPU version reduces once across the block, then copies data back to
the CPU and finishes the reduction there.

The two-stage reduction copies first reduces multiple global elements into local
shared memory, then reduces all elements in shared memory and copies them back
to global memory.  We do this twice to get a single result.
]]


-------------------------------------------------------------------------------
-- Imports                                                                   -- 
-------------------------------------------------------------------------------
import "ebb"

local tid = cudalib.nvvm_read_ptx_sreg_tid_x
local bid_x = cudalib.nvvm_read_ptx_sreg_ctaid_x
local bid_y = cudalib.nvvm_read_ptx_sreg_ctaid_y
local bid_z = cudalib.nvvm_read_ptx_sreg_ctaid_z
local g_dim_x = cudalib.nvvm_read_ptx_sreg_nctaid_x
local g_dim_y = cudalib.nvvm_read_ptx_sreg_nctaid_y
local g_dim_z = cudalib.nvvm_read_ptx_sreg_nctaid_z

local C = require 'compiler.c'

-------------------------------------------------------------------------------
-- Global Parameters                                                         -- 
-------------------------------------------------------------------------------
local BLOCK_SIZE   = 1024
local MAX_GRID_DIM = 65536
local WARP_SIZE    = 32

local N = 100000

-- This symbol can be used to add shared block memory to
-- a terra function intended to run on the GPU.

local rtype = uint32
local treedata = cudalib.sharedmemory(rtype, BLOCK_SIZE)


-------------------------------------------------------------------------------
-- GPU Functions                                                             --
-------------------------------------------------------------------------------
-- generates an unrolled loop of cascading reductions in shared memory. See
-- NVIDIA white paper on reductions for an idea of what this is trying to do:
-- http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
local function unrolled_block_reduce (ptr, tid)
    local expr = quote end
    local step = BLOCK_SIZE

    while (step > 1) do
        step = step/2
        expr = quote
            [expr]
            if tid < step then [ptr][tid] = [ptr][tid] + [ptr][tid + step] end
            cudalib.nvvm_barrier0()
        end

        -- Pairwise reductions over > 32 threads need to be synchronized b/c
        -- they aren't guaranteed to be executed in lockstep, as they are
        -- running in multiple warps.  But the store must be volatile or the
        -- compiler might re-order them!
        --if step > WARP_SIZE then
        --    expr = quote [expr] cudalib.ptx_bar_sync(0) end
        --end
    end
    return expr
end

local u = macro(function(s) return `[uint64](s) end)
local block_id = macro(function() return `u(bid_x()) +
                                          u(bid_y()) * u(g_dim_x()) + 
                                          u(bid_z()) * u(g_dim_y()) * u(g_dim_x()) end)

local terra initialize_array(out : &rtype, max_i : uint64)
    var t : uint64 = tid()
    var b : uint64 = block_id()
    var num_blocks : uint64 = g_dim_x() * g_dim_y() * g_dim_z()

    for i = t + b*u(BLOCK_SIZE), max_i, u(BLOCK_SIZE)*num_blocks do
        out[i] = 1
    end
end

local terra block_reduce(input : &rtype, output : &rtype, N : uint64)
    var t  : uint64 = tid()
    var b  : uint64 = block_id()
    var gt : uint64 = t + b * u(BLOCK_SIZE) -- global index

    -- copy global input into shared memory
    if gt < N then
        treedata[t] = input[gt]
    else
        treedata[t] = 0
    end

    cudalib.ptx_bar_sync(0)
    
    -- reduce chunk in shared memory to one value
    [unrolled_block_reduce(treedata, t)]

    -- copy final value back to global memory
    output[b] = treedata[0]
end

terra two_staged_reduce (input : &rtype, output : &rtype, N : uint64)
    var t  : uint64 = tid()
    var b  : uint64 = block_id()
    var num_blocks : uint64 = g_dim_x() * g_dim_y() * g_dim_z()
    var gt : uint64 = t + b * u(BLOCK_SIZE) -- global index

    -- copy global input into shared memory
    treedata[t] = 0
    for i = gt, N, num_blocks * u(BLOCK_SIZE) do
        treedata[t] = treedata[t] + input[i]
    end

    cudalib.ptx_bar_sync(0)
    
    -- reduce chunk in shared memory to one value
    [unrolled_block_reduce(treedata, t)]

    -- copy final value back to global memory
    output[b] = treedata[0]
end

local R = terralib.cudacompile({initialize_array  = initialize_array,
                                block_reduce      = block_reduce,
                                two_staged_reduce = two_staged_reduce})


-------------------------------------------------------------------------------
-- CPU Functions                                                             -- 
-------------------------------------------------------------------------------
terra get_grid_dimensions (num_blocks : uint64) : {uint, uint, uint}
    if num_blocks < MAX_GRID_DIM then
        return { num_blocks, 1, 1 }
    elseif num_blocks / MAX_GRID_DIM < MAX_GRID_DIM then
        return { MAX_GRID_DIM, (num_blocks + MAX_GRID_DIM - 1) / MAX_GRID_DIM, 1 }
    else
        return { MAX_GRID_DIM, MAX_GRID_DIM, (num_blocks - 1) / MAX_GRID_DIM / MAX_GRID_DIM + 1 }
    end
end

local checkCudaError = macro(function(code)
    return quote
        if code ~= 0 then
            C.printf("CUDA error: ")
            C.printf(C.cudaGetErrorString(code))
            C.printf("\n")
            error(nil)
        end
    end
end)

-------------------------------------------------------------------------------
-- Tests                                                                     -- 
-------------------------------------------------------------------------------
-- Perform a single, block-level reduce on global data, then ship the partially
-- reduce array back to the CPU to finish computation there
terra launch_single_reduction()
    -- allocate device data
    var working_set : &rtype, output: &rtype
    var num_blocks  : uint64 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE

    checkCudaError(C.cudaMalloc([&&opaque](&working_set), sizeof(rtype)*N))
    checkCudaError(C.cudaMalloc([&&opaque](&output),      sizeof(rtype)*N))

    var grid_x : uint, grid_y : uint, grid_z : uint = get_grid_dimensions(num_blocks)
    var launch = terralib.CUDAParams { grid_x, grid_y, grid_z, BLOCK_SIZE, 1, 1, BLOCK_SIZE*sizeof(rtype), nil}

    checkCudaError(R.initialize_array(&launch, working_set, N))
    checkCudaError(R.block_reduce(&launch, working_set, output, N))

    -- copy back partially reduced results
    var result : &rtype = [&rtype](C.malloc(sizeof(rtype)*N))
    checkCudaError(C.cudaMemcpy(result, output, sizeof(rtype)*N, 2))

    -- reduce locally
    var sum : rtype = 0
    for i = 0, num_blocks do
        sum = sum + result[i]
    end

    checkCudaError(C.cudaFree(working_set))
    checkCudaError(C.cudaFree(output))
    C.free(result)

    return sum
end

terra launch_staged_reduction()
    -- allocate device data
    var input : &rtype, output: &rtype
    var num_blocks : uint = (N + BLOCK_SIZE - 1) / BLOCK_SIZE

    checkCudaError(C.cudaMalloc([&&opaque](&input),  sizeof(rtype)*N))
    checkCudaError(C.cudaMalloc([&&opaque](&output), sizeof(rtype)*num_blocks))

    var grid_x : uint, grid_y : uint, grid_z : uint = get_grid_dimensions(num_blocks)
    var launch = terralib.CUDAParams { grid_x, grid_y, grid_z, BLOCK_SIZE, 1, 1, BLOCK_SIZE*sizeof(rtype), nil }
    R.initialize_array(&launch, input, N)

    var remaining_threads : uint64 = N
    while remaining_threads > 1 do
        var remaining_blocks = (remaining_threads + BLOCK_SIZE - 1) / BLOCK_SIZE
        grid_x, grid_y, grid_z = get_grid_dimensions(remaining_blocks)
        var launch = terralib.CUDAParams { grid_x, grid_y, grid_z, BLOCK_SIZE, 1, 1, BLOCK_SIZE*sizeof(rtype), nil }

        R.block_reduce(&launch, input, output, remaining_threads)

        -- Swap input, output
        var tmp = input
        input   = output
        output  = input

        remaining_threads = remaining_blocks
    end

	var result : rtype
	C.cudaMemcpy(&result,output,sizeof(rtype),2)

    C.cudaFree(output)
    C.cudaFree(input)

    return result
end

terra launch_two_staged_reduction ()
    var input : &rtype, partial : &rtype, answer : &rtype
    var num_blocks : uint64 = (N + 8*BLOCK_SIZE - 1) / (BLOCK_SIZE*8)

    checkCudaError(C.cudaMalloc([&&opaque](&input),   sizeof(rtype)*N))
    checkCudaError(C.cudaMalloc([&&opaque](&partial), sizeof(rtype)*num_blocks))
    checkCudaError(C.cudaMalloc([&&opaque](&answer),  sizeof(rtype)*1))

    var grid_x : uint, grid_y : uint, grid_z : uint = get_grid_dimensions(num_blocks)
    var launch  = terralib.CUDAParams { grid_x, grid_y, grid_z, BLOCK_SIZE, 1, 1, 0, nil }
    var launch1 = terralib.CUDAParams { grid_x, grid_y, grid_z, BLOCK_SIZE, 1, 1, BLOCK_SIZE*sizeof(rtype), nil}
    var launch2 = terralib.CUDAParams { 1, 1, 1,                BLOCK_SIZE, 1, 1, BLOCK_SIZE*sizeof(rtype), nil}

    checkCudaError(R.initialize_array(&launch,  input,   N))
    checkCudaError(R.two_staged_reduce(&launch1, input,   partial, N))
    checkCudaError(R.two_staged_reduce(&launch2, partial, answer,  num_blocks))

    var result : rtype
    checkCudaError(C.cudaMemcpy(&result, answer, sizeof(rtype), 2))

    checkCudaError(C.cudaFree(input))
    checkCudaError(C.cudaFree(partial))
    checkCudaError(C.cudaFree(answer))

    return result
end


-------------------------------------------------------------------------------
-- Run script:                                                               -- 
-------------------------------------------------------------------------------
function print_test_result(str, sum, expected, diff)
    if sum == expected then
        print(str .. ' test (size ' .. tostring(N) .. ') executed in ' .. string.gsub(tostring(diff),'ULL','') .. 'ms')
    else
        print(str .. " test FAIL! (Expected " .. tostring(expected) .. ', computed ' .. tostring(sum) .. ')')
    end
end

function test_reduction(fn, name)
    local t0, t1, sum
    t0 = C.clock()
    sum = fn()
    t1 = C.clock()
    print_test_result(name, sum, N, (t1 - t0)*1000/C.CLOCKS_PER_SEC)
end

--test_reduction(launch_single_reduction,     "Single-staged")
--test_reduction(launch_staged_reduction,     "Multi-staged")
test_reduction(launch_two_staged_reduction, "Two-staged")
