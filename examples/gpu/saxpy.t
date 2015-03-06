import 'compiler.liszt'

local tid_x   = cudalib.nvvm_read_ptx_sreg_tid_x
local b_dim_x = cudalib.nvvm_read_ptx_sreg_ntid_x
local bid_x   = cudalib.nvvm_read_ptx_sreg_ctaid_x

local N = 100000


--------------------------------------------------------------------------------
--[[                             liszt saxpy                                ]]--
--------------------------------------------------------------------------------
L.default_processor = L.GPU

function run_liszt_saxpy (tests)
	local R = L.NewRelation { size = N, name = 'R' }
	R:NewField("x", L.float):Load(function(i) return i   end)
	R:NewField("y", L.float):Load(function(i) return i*i end)
	R:NewField("z", L.float)

	local a = L.Global(L.float, 3.49230)

	local liszt kernel liszt_saxpy (r : R)
		r.z = a * r.x + r.y
	end
	
	for i = 1, tests do
		liszt_saxpy(R)
	end
end


--------------------------------------------------------------------------------
--[[                             terra saxpy                                ]]--
--------------------------------------------------------------------------------
local terra terra_saxpy (n : int, a : float , x : &float, y : &float, z : &float)
	var id = tid_x() + b_dim_x() * bid_x()
	if id < n then
		z[id] = a * x[id] + y[id]
	end
end
local saxpy_kernel = terralib.cudacompile({terra_saxpy=terra_saxpy},true).terra_saxpy

function run_terra_saxpy (tests)
	local T = L.NewRelation { size = N, name = 'R' }
	T:NewField("x", L.float):Load(function(i) return i   end)
	T:NewField("y", L.float):Load(function(i) return i*i end)
	T:NewField("z", L.float)

	local terra launch_terra_saxpy (n : int)
		var x : &float = [&float]([T.x:DataPtr()])
		var y : &float = [&float]([T.y:DataPtr()])
		var z : &float = [&float]([T.z:DataPtr()])
		var a : float = 3.49230

		var ts : int = 64
		var bs : int = (n + ts - 1) / ts

		var params = terralib.CUDAParams {bs, 1, 1, ts, 1, 1, 0, nil}
		saxpy_kernel(&params, n, a, x, y, z)
	end
	for i = 1, tests do
		launch_terra_saxpy(N)
	end
end


--------------------------------------------------------------------------------
--[[                              execute                                   ]]--
--------------------------------------------------------------------------------
run_liszt_saxpy(1)
run_terra_saxpy(1)

