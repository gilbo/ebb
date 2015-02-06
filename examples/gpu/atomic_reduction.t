if not terralib.cudacompile then error("Terra is not compiled with cuda support") end

terralib.linklibrary("/usr/local/cuda/lib/libcuda.dylib")
--local cas32 = terralib.intrinsic('llvm.nvvm.atom.cas.gen.i', {&int, int, int} -> {int})

local C = require 'compiler.c'
local aadd = terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32", {&float,float} -> {float})

terra reduce_float (sum : &float)
	aadd(sum, 1)
	var test : int = 1
--	cas32(&test, 1, 0)
end

local R = terralib.cudacompile { reduce_float = reduce_float }
local nReductions = 1024

terra reduce_test() : float
	var reducef : float = 0.0
	var dreducef : &float
	var launch = terralib.CUDAParams { 1, 1, 1, nReductions, 1, 1, 0, nil}

	-- allocate reduction variable on GPU & copy over value
	C.cudaMalloc([&&opaque](&dreducef),
		         sizeof(float))
	C.cudaMemcpy([&opaque](dreducef),
		         [&opaque](&reducef),
		         sizeof(float),
		         C.cudaMemcpyHostToDevice)

	R.reduce_float(&launch, dreducef)

	C.cudaMemcpy([&opaque](&reducef),
		         [&opaque](dreducef),
		         sizeof(float),
		         C.cudaMemcpyDeviceToHost)
	C.cudaFree(dreducef)
	return reducef
end

local BLOCK_SIZE = 32
local thread_id  = cudalib.nvvm_read_ptx_sreg_tid_x
local block_id   = cudalib.nvvm_read_ptx_sreg_ctaid_x

params = 