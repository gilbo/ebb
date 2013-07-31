-- Link in C++ implementation of single core runtime
module('runtime', package.seeall)

local runtime = terralib.includec("runtime/common/liszt_runtime.h")

local osf = assert(io.popen('uname', 'r'))
local osname = assert(osf:read('*l'))
osf:close()

if osname == 'Linux' then
	terralib.linklibrary("runtime/single/libruntime_single.so")
elseif osname == 'Darwin' then
	terralib.linklibrary("runtime/single/runtime_single.dylib")
else
	error("Unknown Operating System")
end

-- lELementTypes
L_VERTEX = 0
L_CELL   = 1
L_EDGE   = 2
L_FACE   = 3

-- lTypes
L_INT    = 0
L_FLOAT  = 1
L_DOUBLE = 2
L_BOOL   = 3
L_STRING = 4

-- lReductions
L_ASSIGN   = 0
L_PLUS     = 1
L_MINUS    = 2
L_MULTIPLY = 3
L_DIVIDE   = 4
L_MIN      = 5
L_MAX      = 6
L_BOR      = 7
L_BAND     = 8
L_XOR      = 9
L_AND      = 10
L_OR       = 11

-- lPhase
L_READ_ONLY       = 0
L_WRITE_ONLY      = 1
L_MODIFY          = 2
L_REDUCE_PLUS     = 3
L_REDUCE_MULTIPLY = 4
L_REDUCE_MIN      = 5
L_REDUCE_MAX      = 6
L_REDUCE_BOR      = 7
L_REDUCE_BAND     = 8
L_REDUCE_XOR      = 9
L_REDUCE_AND      = 10
L_REDUCE_OR       = 11

-- structs used in liszt kernels/stencil functions
lContext        = runtime.lContext
lsFunctionTable = runtime.lsFunctionTable
lsElement       = runtime.lsElement
lSet            = runtime.lSet
lStencilData    = runtime.lStencilData


lkContext       = runtime.lkContext
lkElement       = runtime.lkElement

-- functions called directly
lkGetActiveElement = runtime.lkGetActiveElement
lkFieldRead        = runtime.lkFieldRead
lkScalarWrite      = runtime.lkScalarWrite
lScalarWrite       = runtime.lScalarWrite
lVerticesOfMesh    = runtime.lVerticesOfMesh
lFieldEnterPhase   = runtime.lFieldEnterPhase
lScalarEnterPhase  = runtime.lScalarEnterPhase
lKernelRun         = runtime.lKernelRun
lNewlSet           = runtime.lNewlSet
lFreelSet          = runtime.lFreelSet

struct Scalar {
   lscalar  : &runtime.lScalar;
}

terra Scalar:lScalar ( )
   return self.lscalar
end

terra Scalar:lkScalar ( )
   return self.lScalar.lkscalar;
end


struct Mesh {
	ctx : &lContext;
	nVertices : int;
	nEdges    : int;
	nFaces    : int;
	nCells    : int;
}

terra loadMesh (filename : rawstring) : { &lContext }
	var ctx : &lContext = runtime.lLoadContext(filename)
	return ctx
end

--[[
function loadMesh (filename)
	return runtime.lLoadContext(filename)
end
]]
function loadField (mesh, name, key_type, val_type, val_length)
	return runtime.lLoadField(mesh.ctx,name,L_VERTEX,L_FLOAT,val_length)
end

function initField(mesh, key_type, val_type, val_length)
	return runtime.lInitField(mesh.ctx, key_type, val_type, val_length)
end

terra initScalar(ctx : &runtime.lContext, val_type : uint, val_length : uint)
	var s = Scalar { runtime.lInitScalar(ctx, L_FLOAT, 3) }
	return s
end

-- function lScalarWrite(mesh, scalar, reduction, data_type, data_length, value_offset, value_length, value)
