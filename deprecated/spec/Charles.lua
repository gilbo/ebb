
Constants = { }
Constants.iterations = 10

IC = {
	-- simulation properties
	cfl          = 0.5,
	dt           = 0.5,
	force_x      = 1.0,
	-- gas properties
	gamma        = 1.4,
	p_ref        = 7100,
	rho_ref      = 1.0,
	T_ref        = 1.0,
	R_gas        = 0.0,
	mu_ref       = 0.00253165,
	mu_power_law = 0.0,
	Pr_lam       = 0.7,
	-- output
	check_interval = 10	
}

ChannelCf4 = {
	dt   = 0.0,
	time = 0.0, -- simulation time
	step = 0,
	
	rho  = LisztField[Cell, Float](0),
	rhou = LisztField[Cell, Float3](LisztConstants.float3_zero),
	rhoE = LisztField[Cell, Float](0),
	
	mu_lam = LisztField[Cell, Float](0),
	cp     = LisztField[Cell, Float](0),
	k_lam  = LisztField[Cell, Float](0),
	
	u = LisztField[Cell, Float3](LisztConstants.float3_zero),
	p = LisztField[Cell, Float](0),
	T = LisztField[Cell, Float](0),
	
	dil = LisztFieldWithConst[Cell, Float](0),
	
	mu_sgs = LisztField[Cell, Float](0),
	k_sgs  = LisztField[Cell, Float](0),
	
	fa_alpha = FLisztieldWithConst[Face, Float](0),

	-- data req'd by the RK solver
	drho0  = LisztField[Cell, Float](0),
	drhou0 = LisztField[Cell, Float3](LisztConstants.float3_zero),
	drhoE0 = LisztField[Cell, Float](0),
	drho1  = LisztField[Cell, Float](0),
	drhou1 = LisztField[Cell, Float3](LisztConstants.float3_zero),
	drhoE1 = LisztField[Cell, Float](0),
	drho2  = LisztField[Cell, Float](0),
	drhou2 = LisztField[Cell, Float3](LisztConstants.float3_zero),
	drhoE2 = LisztField[Cell, Float](0)
}


ChannelCf4.turboChannelIC = function () {
	local kernel = liszt_kernel (cell) {
		local center = MeshGeometry.x_cv(cell)

		local rhou_me = LisztVec(IC.rho_ref * 1.5 * (1 + center.y*center.y) * (1 - center.y*center.y), 0, 0)
		local eps     = .05 * rhou_me.x
		rhou_me      += LisztVec(eps, eps, eps)

		ChannelCf4.rho(cell)  = IC.rho_ref
		ChannelCf4.rhou(cell) = rhou_me
		ChannelCf4.rhoE(cell) = IC.p_ref / (IC.gamma - 1) + .5 * dot(rhou_me, rhou_me) / IC.rho_ref
	}
	mesh.cells().map(kernel)
}

ChannelCf4.init = function () {
	print("LISZT HOOK: INIT")
	local cells = mesh.cells()
	MeshGeometry.calcGeometry()

	ChannelCf4.initialHook()
	ChannelCf4.updateConservativeData()
	ChannelCf4.updatePrimitiveData(ChannelCf4.time)
	ChannelCf4.initProbes()
}

ChannelCf4.initialHook = function () {
	print("ChannelCf4: initialHook()")
	ChannelCf4.turbChannelIC()
}

ChannelCf4.updatePrimitiveData = function (t) {
	printf("updatePrimitiveData(t): IMPLEMENT ME")
}

ChannelCf4.updateConservativeData = function () {}
ChannelCf4.initProbes             = function () {}
ChannelCf4.updateStats            = function () {}

ChannelCf4.run = function () {
	print("LISZT HOOK: RUN")

	ChannelCf4.doChecks()
	local done = ChannelCf4.doneSolver()
	while (done != true)
		ChannelCf4.step += 1
		ChannelCf4.calcDt()
		ChannelCf4.time	+= ChannelCf4.dt

		if (IC.check_interval > 0) and (ChannelCf4.step % IC.check_interval == 0) then
			print("----------------------------------------------------------")
			print(" starting step: ", ChannelCf4.step, " time: ", ChannelCf4.time, " dt: ", ChannelCf4.dt);
			print("----------------------------------------------------------")
		end

		ChannelCf4.calcSgsStuff(step % IC.check_interval == 0)

		-- rk step 1...
		ChannelCf4.calcRhs()
		
		ChannelCf4.updatePrimitiveData(ChannelCf4.time)

		-- rk step 2...
		ChannelCf4.calcRhs()

		local update_drhs = liszt_kernel (cell) {
			local tmp = ChannelCf4.dt / MeshGeometry.cv_volume(cell)
			ChannelCf4.drho0(cell)  *= tmp
			ChannelCf4.drhou0(cell) *= LisztVec(tmp, tmp, tmp)
			ChannelCf4.drhoE0(cell) *= tmp
		}	
		mesh.cells().map(update_drhs)

		updatePrimitiveData(time - .4*ChannelCf4.dt)

		-- rk step 3...
		ChannelCf4.calcRhs()
		ChannelCf4.updatePrimitiveData(ChannelCf4.time)

		if ChannelCf4.step % IC.check_interval == 0 then
			doChecks()
			done = doneSolver()
		end
	end
}

ChannelCf4.calcDt    = function () { print("calcDT(): IMPLEMENT ME") }
ChannelCf4.calcDT(b) = function () { print("calcSgsStuff(): IMPLEMENT ME") }

ChannelCf4.doneSolver = function () { 
	return Constants.iterations >= 0 and 
	       ChannelCf4.step      >= Constants.iterations 
}

ChannelCf4.doChecks = function () {
	-- do some stuff...
	print("doChecks(): IMPLEMENT ME")
}

