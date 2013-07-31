terralib.require("runtime/liszt")
std = terralib.includec("stdio.h")
local runtime = runtime
require "include/liszt"

mesh     = LoadMesh("examples/mesh.lmesh")
position = mesh:fieldWithLabel(Vertex, Vector.type(float,3), "position")

local function main ( )
	-- initialize com...
	local com = mesh:scalar(Vector.type(float,3))
	local terra init_com ()
		var tmp : float[3]
		tmp[0] = 0.0
		tmp[1] = 0.0
		tmp[2] = 0.0
		runtime.lScalarWrite(mesh.ctx, [com:lScalar()], runtime.L_ASSIGN, runtime.L_FLOAT, 3,0,3,&tmp)
	end
	init_com()

	local terra sum_pos_kernel (ctx_struct : runtime.lkContext)
		var ctx : &runtime.lkContext = &ctx_struct
		var v   : runtime.lkElement
		if (runtime.lkGetActiveElement(ctx, &v) > 0) then
			var temp5 : float[3]
			runtime.lkFieldRead([position:lkField()], v, runtime.L_FLOAT,3,0,3,&temp5)
			runtime.lkScalarWrite(ctx, [com:lkScalar()], runtime.L_PLUS, runtime.L_FLOAT,3,0,3,&temp5)
			-- std.printf("%f %f %f\n", temp5[0], temp5[1], temp5[2])
		end
	end

	--[[ this has been generated for iterating over vertices... 
	     but in a liszt kernel, we won't always know what set we 
	     are iterating over until the user calls the kernel 
	--]]
	local terra run_sum_pos_kernel (ctx : &runtime.lContext)
		var temp7 : &runtime.lSet = runtime.lNewlSet()
		runtime.lVerticesOfMesh(ctx,temp7)
		runtime.lKernelRun(ctx, temp7, runtime.L_VERTEX, 0, sum_pos_kernel)
		runtime.lFreelSet(temp7)
	end

	run_sum_pos_kernel(mesh.ctx)

	local terra print_result (ctx : & runtime.lContext)
		var temp9 : &runtime.lSet = runtime.lNewlSet()
		runtime.lVerticesOfMesh(ctx, temp9)
		var temp10 : int = runtime.lSetSize(ctx, temp9)
		runtime.lFreelSet(temp9)

		runtime.lScalarEnterPhase([com:lScalar()], runtime.L_FLOAT, 3, runtime.L_READ_ONLY)
		var tmp_1 : float[3]
		tmp_1[0] = temp10
		tmp_1[1] = temp10
		tmp_1[2] = temp10

		runtime.lScalarWrite(ctx, [com:lScalar()], runtime.L_DIVIDE, runtime.L_FLOAT, 3,0,3,&tmp_1)
		var tmp_2 : float[3];
		runtime.lScalarRead(ctx, [com:lScalar()], runtime.L_FLOAT, 3,0,3,&tmp_2)

		std.printf("final... %f %f %f\n", tmp_2[0], tmp_2[1], tmp_2[2])
	end

	print_result(mesh.ctx)
end

main()