-- Link in C++ implementation of single core runtime
module('runtime', package.seeall)

local runtime = terralib.includec("runtime/common/liszt_runtime.h")
terralib.linklibrary("runtime/single/runtime_single.dylib")

lr = runtime

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


terra loadMesh (filename : rawstring)
	var ctx = runtime.lLoadContext(filename)
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
	return runtime.lInitField(mesh.ctx, mesh.num_fields, key_type, val_type, val_length)
end

