-- Link in C++ implementation of single core runtime

module('runtime', package.seeall)

local runtime = terralib.includec("runtime/single/liszt_runtime.h")

function link_runtime ()
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
end
link_runtime()

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

lkElement       = runtime.lkElement
lkContext       = runtime.lkContext

lsContext       = runtime.lsContext
lsFunctionTable = runtime.lsFunctionTable
lsElement       = runtime.lsElement
lSet            = runtime.lSet
lStencilData    = runtime.lStencilData


-- functions called directly
lScalarRead        = runtime.lScalarRead
lScalarWrite       = runtime.lScalarWrite
lVerticesOfMesh    = runtime.lVerticesOfMesh
lFieldEnterPhase   = runtime.lFieldEnterPhase
lScalarEnterPhase  = runtime.lScalarEnterPhase
lKernelRun         = runtime.lKernelRun
lNewlSet           = runtime.lNewlSet
lFreelSet          = runtime.lFreelSet
lSetSize           = runtime.lSetSize

lkGetActiveElement = runtime.lkGetActiveElement
lkFieldRead        = runtime.lkFieldRead
lkScalarWrite      = runtime.lkScalarWrite


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

terra loadField (ctx : &runtime.lContext, name : &int8, key_type : uint, val_type : uint, val_length : uint)
    return runtime.lLoadField(ctx,name,L_VERTEX,L_FLOAT,val_length)
end

function initField(ctx, key_type, val_type, val_length)
	return runtime.lInitField(ctx, key_type, val_type, val_length)
end

terra getlkField (field : &runtime.lField)
	return field.lkfield
end

terra initScalar(ctx : &runtime.lContext, val_type : uint, val_length : uint)
	return runtime.lInitScalar(ctx, val_type, val_length)
end

terra getlkScalar (scalar : &runtime.lScalar)
	return scalar.lkscalar
end
