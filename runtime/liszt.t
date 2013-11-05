-- Link in C++ implementation of single core runtime

local R = {}

local runtime = terralib.includec("runtime/single/liszt_runtime.h")
local util    = terralib.require("runtime/util")
util.link_runtime()

-- lELementTypes
R.L_VERTEX = 0
R.L_CELL   = 1
R.L_EDGE   = 2
R.L_FACE   = 3

-- lTypes
R.L_INT    = 0
R.L_FLOAT  = 1
R.L_DOUBLE = 2
R.L_BOOL   = 3
R.L_STRING = 4

-- structs used in liszt kernels/stencil functions
local lContext = runtime.lContext
R.lContext     = lContext

-- parameter types
local lType        = uint
local lElementType = uint
local size_t       = uint
local lReduction   = uint

terra R.loadBoundarySet (ctx : &lContext, typ : lElementType, boundary_name : rawstring)
	var size : uint64
	var data = runtime.lLoadBoundarySet(ctx, typ, boundary_name, &size)
	return data, size
end

terra R.loadMesh (filename : rawstring) : { &lContext }
	var ctx : &lContext = runtime.lLoadContext(filename)
	return ctx
end

return R