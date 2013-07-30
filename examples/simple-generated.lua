terralib.require("runtime/liszt")
local runtime = runtime
require "include/liszt"

mesh     = LoadMesh("examples/mesh.lmesh")
position = mesh:fieldWithLabel(Vertex, Vector.type(float,3), "position")
field    = mesh:field(Face, float, 0.0)

local function main ( )
	local com = mesh:scalar(Vector.type(float,3))
	local terra init_com ()
		var tmp : float[3]
		tmp[0] = 0.0
		tmp[1] = 0.0
		tmp[2] = 0.0
		runtime.lScalarWrite(mesh.ctx, com:lScalar(), runtime.L_ASSIGN, runtime.L_FLOAT, 3,0,3,&tmp)
	end
	init_com()

	local terra sum_pos_stencil (tbl: &runtime.lsFunctionTable, ctx :&runtime.lContext) : { }
		var e0 : runtime.lsElement
		tbl.lsGetActiveElement(ctx, &e0);
		tbl.lsFieldAccess(ctx,0,runtime.L_VERTEX,&e0,runtime.L_READ_ONLY)
		return
	end

	local terra sum_pos_kernel_stencil_data ()
		var data : runtime.lStencilData;
		data.stencil_function = sum_pos_stencil
		data.is_trivial = true
		return data
	end

	local terra sum_pos_kernel (ctx_struct : runtime.lkContext)
		var ctx : &runtime.lkContext = &ctx_struct
		var v   : &runtime.lkElement
		if (runtime.lkGetActiveElement(ctx, &v)) then
			var temp5 : float[3]
	
			runtime.lkFieldRead(position:lkField(), v, runtime.L_FLOAT,3,0,3,&temp5)
			runtime.lkScalarWrite(ctx, com:lkScalar(), runtime.L_PLUS, runtime.L_FLOAT,3,0,3,&temp5)
		end
	end

	local terra run_sum_pos_kernel (ctx : &runtime.lContext)
		var temp7 : &runtime.lSet = runtime.lNewlSet()
		runtime.lVerticesOfMesh(ctx,temp7)
		runtime.lScalarEnterPhase(com:lScalar(), runtime.L_FLOAT, 3, runtime.L_REDUCE_PLUS)
		runtime.lFieldEnterPhase(position:lField(), runtime.L_FLOAT, 3, runtime.L_READ_ONLY)
		runtime.lKernelRun(ctx, temp7, runtime.L_VERTEX, 0, sum_pos_kernel, sum_pos_kernel_stencil_data())
		runtime.lFreelSet(temp7)
	end

	run_sum_pos_kernel(mesh.ctx)


end

main()