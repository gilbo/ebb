import Liszt.Language._
import Liszt.MetaInteger._

@lisztcode
object Constants {
	/************************************************************************************/
	/* CONSTANTS */
	/************************************************************************************/
	val DEBUG_FLAG = true
	// val DEBUG_FLAG = false
	val VolumeError = -1
	val QStopError = -2
}

@lisztcode
object Mesh {

	/************************************************************************************/
	/* MESH PROPERTIES */
	/************************************************************************************/
	val numElem : Int = size(cells(mesh)) /* Elements/Nodes in this domain */
	val numNode : Int = size(vertices(mesh))
	val edgeNodes : Int = round(cbrt(numNode)).toInt;
	val edgeElems : Int = round(cbrt(numElem)).toInt;

	/************************************************************************************/
	/* ELEMENT-CENTERED TEMPORARY PROPERTIES */
	/************************************************************************************/
	// Principal strains (x,y,z) temp
	// Velocity gradient (x,y,z) temp
	// Coordinate gradient (x,y,z) temp
	// These use scratchpadexx

	// New relative volume temp
	val vnew = FieldWithConst[Cell,Double](0.0)

	// Minimum index for each cell's neighboring cell
	// TODO: This will not be required once Liszt supports cell orientation
	val minIdx = FieldWithConst[Cell,Int](numNode+1)

	/************************************************************************************/
	/* OTHER TEMPORARY PROPERTIES */
	/************************************************************************************/
	// TODO: Define fields over arbitrary sets of elements (cells(mesh))
	// For elements
	// var padNum = 0
	val scratchpade01 = FieldWithConst[Cell,Double](0.0)
	val scratchpade02 = FieldWithConst[Cell,Double](0.0)
	val scratchpade03 = FieldWithConst[Cell,Double](0.0)
	val scratchpade04 = FieldWithConst[Cell,Double](0.0)
	val scratchpade05 = FieldWithConst[Cell,Double](0.0)
	val scratchpade06 = FieldWithConst[Cell,Double](0.0)

	var start_time = 0.0
	var end_time = 0.0

	/************************************************************************************/
	/* NODE-CENTERED PROPERTIES */
	/************************************************************************************/
	// coordinates (x,y,z)
	val position = FieldWithLabel[Vertex,Vec[_3,Double]]("position")

	// velocities (x,y,z)
	val velocity = FieldWithConst[Vertex, Vec[_3,Double]](Vec(0.0, 0.0, 0.0))

	// TODO: Comment these three fields for implementing
	// loop fusion on CalcPosition
	//val xdd = FieldWithConst[Vertex,Double](0.0)
	//val ydd = FieldWithConst[Vertex,Double](0.0)
	//val zdd = FieldWithConst[Vertex,Double](0.0)

	// forces (x,y,z)
	val forces = FieldWithConst[Vertex,Vec[_3,Double]](Vec(0.0, 0.0, 0.0))

	// node mass (m)
	val nodalMass = FieldWithConst[Vertex,Double](0.0)

	// TODO: Comment to implement loop fusion on CalcPosition
	//val symmX = BoundarySet[Vertex]("symmX")
	//val symmY = BoundarySet[Vertex]("symmY")
	//val symmZ = BoundarySet[Vertex]("symmZ")
	// TODO: Uncomment to implement loop fusion on CalcPosition
	val xsymm = BoundarySet[Vertex]("symmX")
	val ysymm = BoundarySet[Vertex]("symmY")
	val zsymm = BoundarySet[Vertex]("symmZ")
	val symmX = FieldWithConst[Vertex, Int](1)
	val symmY = FieldWithConst[Vertex, Int](1)
	val symmZ = FieldWithConst[Vertex, Int](1)

	/************************************************************************************/

	/************************************************************************************/
	/* ELEMENT-CENTERED PROPERTIES */
	/************************************************************************************/
	// Sets of cells, each corresponding to a different material. All cells same material for now
	val material = BoundarySet[Cell]("material")

	// Energy
	val e = FieldWithConst[Cell,Double](0.0)

	// Pressure 
	val p = FieldWithConst[Cell,Double](0.0)

	// q
	val q = FieldWithConst[Cell,Double](0.0)

	// Linear term for q
	val ql = FieldWithConst[Cell,Double](0.0)

	// Quadratic term for q
	val qq = FieldWithConst[Cell,Double](0.0)

	// Relative volume 
	val v = FieldWithConst[Cell,Double](1.0)

	// Reference volume 
	val volo = FieldWithConst[Cell,Double](0.0)

	// vnew - v
	val delv = FieldWithConst[Cell,Double](0.0)

	// Volume derivative over volume
	val vdov = FieldWithConst[Cell,Double](0.0)

	// Characteristic length of an element
	val arealg = FieldWithConst[Cell,Double](0.0)

	// Sound speed 
	val ss = FieldWithConst[Cell,Double](0.0)

	// Element Mass
	val elementMass = FieldWithConst[Cell,Double](0.0)

	// lxim, lxip, letam, letap, lzetam, lzetap
	val neighborIndexes = FieldWithConst[Cell,Vec[_6, Int]](Vec(0, 0, 0, 0, 0, 0))

	/************************************************************************************/

	/* GLOBAL MESH PARAMETERS */
	val	dtfixed : Double = -1.0e-7			/* fixed time increment */
	var time : Double	= 0.0				/* current time */
	var deltatime : Double = 0.0			/* variable time increment */
	val	deltatimemultlb : Double = 1.1
	val	deltatimemultub : Double = 1.2
	val stoptime : Double = 1.0e-2			/* end time for simulation */
	var cycle : Int = 0						/* iteration count for simulation */

	val hgcoef : Double = 3.0				/* hourglass control */
	val	qstop : Double = 1.0e+12			/* excessive q indicator */
	val monoq_max_slope : Double = 1.0
	val monoq_limiter_mult : Double = 2.0
	val	e_cut : Double = 1.0e-7				/* energy tolerance */
	val p_cut : Double = 1.0e-7				/* pressure tolerance */
	val	u_cut : Double = 1.0e-7				/* velocity tolerance */
	val	q_cut : Double = 1.0e-7				/* q tolerance */
	val v_cut : Double = 1.0e-10			/* relative volume tolerance */
	val qlc_monoq : Double = 0.5			/* linear term coef for q */
	val qqc_monoq : Double = 2.0/3.0		/* quadratic term coef for q */
	val qqc : Double = 2.0
	val	eosvmax : Double = 1.0e+9
	val	eosvmin : Double = 1.0e-9
	val pmin : Double = 0.0					/* pressure floor */
	val	emin : Double = -1.0e+15			/* energy floor */
	val dvovmax : Double = 0.1				/* maximum allowable volume change */
	val refdens : Double = 1.0				/* reference density */

	var dtcourant : Double = 0.0			/* courant constraint */
	var dthydro : Double = 0.0				/* volume change constraint */
	val	dtmax : Double = 1.0e-2				/* maximum allowable time increment */


	// TODO : Add dimension length in Liszt
	var sizeX : Int = 0						/* X,Y,Z extent of this block */
	var sizeY : Int = 0
	var sizeZ : Int = 0

	var timeCourant   : Double = 0.0
	var timeHydro     : Double = 0.0
	var timePosition  : Double = 0.0
	var timeUpdateVol : Double = 0.0
	var timeIntegrateStress : Double = 0.0
	var timeHourglass : Double = 0.0
	var timeKinQ : Double = 0.0
	var timeQRegionEOS : Double = 0.0

	def initMeshParameters() {

		/*********************************************************************************************/
		/* INIT GLOBAL PARAMETERS */
		/*********************************************************************************************/

		deltatime = 1.0e-7
		time  = 0.0
		cycle = 0

		dtcourant = 1.0e+20
		dthydro = 1.0e+20

		/*********************************************************************************************/

		/*********************************************************************************************/
		/* INIT FIELD VALUES */
		/*********************************************************************************************/
		// Calc local indexes for cell vertices
		BuildMeshOrientation()

		// TODO : Create local arrays, currently accessing everything thru "vertices(cell)"
		/* initialize field data */
		for (c <- cells(mesh)) {
			// volume calculations
			val localCoords = getLocalNodeCoordVectors(c)
			var volume = CalcElemVolume(localCoords)
			volo(c) = volume
			elementMass(c) = volume
			for (v <- vertices(c)) {
				nodalMass(v) += volume/8.0
			}
		}

		/* deposit energy */
		// TODO : Define origin node, to avoid iterating all nodes unnecessarily
		// Initialize energy for one node
		for(c <- cells(mesh)) {
			if(ID(c) == 1) {
				e(c) = 3.948746e+7
			}
		}

		for(c <- cells(mesh)) {
			for (v <- vertices(c)) {
				minIdx(c) = minIdx(c) min ID(v)
			}
		}

		//TODO: Uncomment next three for loops to implement loop fusion on CalcPosition
		for(v <- xsymm) {
			symmX(v) = 0
		}
		for(v <- ysymm) {
			symmY(v) = 0
		}
		for(v <- zsymm) {
			symmZ(v) = 0
		}
	}

	def TimeIncrement() {
		var targetdt = stoptime - time

		if((dtfixed <= 0.0) && (cycle != 0)) {
			val olddt = deltatime 

			/* This will require a reduction in parallel */
			var newdt = 1.0e+20
			if (dtcourant < newdt) {
				newdt = dtcourant / 2.0
			}
			if (dthydro < newdt) {
				newdt = dthydro * 2.0 / 3.0
			}

			var ratio = newdt / olddt
			if (ratio >= 1.0) {
				if (ratio < deltatimemultlb) {
					newdt = olddt
				}
				else if (ratio > deltatimemultub) {
					newdt = olddt*deltatimemultub
				}
			}

			val dtmax_tmp = dtmax
			if (newdt > dtmax_tmp) {
				newdt = dtmax_tmp
			}
			deltatime = newdt
		}

		val dttmp = deltatime

		/* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
		if ((targetdt > dttmp) && (targetdt < (4.0 * dttmp / 3.0))) {
			targetdt = 2.0 * dttmp / 3.0
		}

		if (targetdt < dttmp) {
			deltatime = targetdt
		}
		time += deltatime
		cycle += 1
	}

	def CalcElemShapeFunctionDerivatives1(localCoords: Mat[_8, _3, Double]) : Double = {
	
		val r0 = row(localCoords, 0)
		val r1 = row(localCoords, 1)
		val r2 = row(localCoords, 2)
		val r3 = row(localCoords, 3)
		val r4 = row(localCoords, 4)
		val r5 = row(localCoords, 5)
		val r6 = row(localCoords, 6)
		val r7 = row(localCoords, 7)

		val r60 = r6-r0
		val r53 = r5-r3
		val r71 = r7-r1
		val r42 = r4-r2
		val fjet = r60 - r53 + r71 - r42;
	
		val r6053 = r60 + r53
		val r7142 = r71 + r42	
		val fjxi = r6053 - r7142;
		val fjze = r6053 + r7142;
	
		val cjet = cross(fjze, fjxi);

		/* calculate jacobian determinant (volume) */
		return 8.0 * 0.125 * 0.125 * 0.125 * dot(fjet, cjet)
	}

	// TODO: FIX the nasty HACK to return B and detJ together
	// Currently bundled as a single vector of multiple values
	def CalcElemShapeFunctionDerivatives2(localCoords: Mat[_8, _3, Double]) : Mat[_9, _3, Double] = {
	
		val r0 = row(localCoords, 0)
		val r1 = row(localCoords, 1)
		val r2 = row(localCoords, 2)
		val r3 = row(localCoords, 3)
		val r4 = row(localCoords, 4)
		val r5 = row(localCoords, 5)
		val r6 = row(localCoords, 6)
		val r7 = row(localCoords, 7)
	
		val r60 = r6 - r0
		val r53 = r5 - r3
		val r71 = r7 - r1
		val r42 = r4 - r2
		val fjet = 0.125 * (r60 - r53 + r71 - r42);

		val r6053 = r60 + r53
		val r7142 = r71 + r42	
		val fjxi = 0.125 * (r6053 - r7142);
		val fjze = 0.125 * (r6053 + r7142);
	
		val cjxi = cross(fjet, fjze);
		val cjet = cross(fjze, fjxi);
		val cjze = cross(fjxi, fjet);

		/* calculate partials :
		   this need only be done for l = 0,1,2,3   since , by symmetry ,
		   (6,7,4,5) = - (0,1,2,3) .
		 */
		val temp0 = cjxi + cjet
		val temp1 = cjxi - cjet

		val b0 = - temp0 - cjze;
		val b1 =   temp1 - cjze;
		val b2 =   temp0 - cjze;
		val b3 = - temp1 - cjze;
		val b4 = - b2;
		val b5 = - b3;
		val b6 = - b0;
		val b7 = - b1;
	
		/* calculate jacobian determinant (volume) */
		val volume = 8.0 * dot(fjet, cjet)
		// TODO: HACK HACK HACK!!!
		val b8 : Vec[_3, Double] = Vec(volume, 0.0, 0.0)
		return Mat(b0, b1, b2, b3, b4, b5, b6, b7, b8)
	}
	
	def SumElemFaceNormal(coords20 : Vec[_3, Double], coords31 : Vec[_3, Double], stress : Vec[_3, Double]) : Vec[_3, Double] = {
		val bisect0 = coords20 + coords31;
		val bisect1 = coords20 - coords31;
		val area = -0.0625 * cross(bisect0, bisect1) * stress;
		return area;
	}
	
	def CalcElemNodeNormals(localCoords : Mat[_8, _3, Double], stress : Vec[_3, Double]) : Mat[_8, _3, Double] = {
	
		val r0 = row(localCoords, 0)
		val r1 = row(localCoords, 1)
		val r2 = row(localCoords, 2)
		val r3 = row(localCoords, 3)
		val r4 = row(localCoords, 4)
		val r5 = row(localCoords, 5)
		val r6 = row(localCoords, 6)
		val r7 = row(localCoords, 7)
	
		/* evaluate face one: nodes 0, 1, 2, 3 */
		val temp0 = SumElemFaceNormal(r2-r0, r3-r1, stress);
	
		/* evaluate face two: nodes 0, 4, 5, 1 */
		val temp1 = SumElemFaceNormal(r5-r0, r1-r4, stress);
	
		/* evaluate face three: nodes 1, 5, 6, 2 */
		val temp2 = SumElemFaceNormal(r6-r1, r2-r5, stress);
	
		/* evaluate face four: nodes 2, 6, 7, 3 */
		val temp3 = SumElemFaceNormal(r7-r2, r3-r6, stress);
	
		/* evaluate face five: nodes 3, 7, 4, 0 */
		val temp4 = SumElemFaceNormal(r4-r3, r0-r7, stress);
	
		/* evaluate face six: nodes 4, 7, 6, 5 */
		val temp5 = SumElemFaceNormal(r6-r4, r5-r7, stress);
	
		val pf = Mat(temp0+temp1+temp4, temp0+temp1+temp2, temp0+temp2+temp3, temp0+temp3+temp4, temp1+temp4+temp5, temp1+temp2+temp5, temp2+temp3+temp5, temp3+temp4+temp5);
	
		return pf;
	}

	// Eliminated call to SumElemStressesToNodeForces; performing stress multiplication inside SumElemFaceNormal	

	def IntegrateStressForElems() {
		val determ = scratchpade01;

		for (c <- cells(mesh)) {
			val localCoords = getLocalNodeCoordVectors(c)
			/* Volume calculation involves extra work for numerical consistency. */
			determ(c) = CalcElemShapeFunctionDerivatives1(localCoords)

			val stress = -p(c)-q(c)
			val f = CalcElemNodeNormals(localCoords, Vec(stress, stress, stress));
	
			//TODO: Change according to order
			val v0 = vertex(c,0)
			forces(v0) += row(f,0)
	
			val v1 = vertex(c,1)
			forces(v1) += row(f,1)
	
			val v2 = vertex(c,2)
			forces(v2) += row(f,2)
	
			val v3 = vertex(c,3)
			forces(v3) += row(f,3)
	
			val v4 = vertex(c,4)
			forces(v4) += row(f,4)
	
			val v5 = vertex(c,5)
			forces(v5) += row(f,5)
	
			val v6 = vertex(c,6)
			forces(v6) += row(f,6)
	
			val v7 = vertex(c,7)
			forces(v7) += row(f,7)
		}
	}
	
	def VoluDer(x0 : Double, x1 : Double, x2 : Double, x3 : Double, x4 : Double, x5 : Double,
			y0 : Double, y1 : Double, y2 : Double, y3 : Double, y4 : Double, y5 : Double,
			z0 : Double, z1 : Double, z2 : Double, z3 : Double, z4 : Double, z5 : Double) : Vec[_3, Double] = {
		//TODO: Switch to vector operations if feasible
		val dvdx =
				(y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
				(y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
				(y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
		val dvdy =
				- (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
				(x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
				(x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);
	
		val dvdz =
			   - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
			   (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
			   (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);
		return Vec(dvdx/12.0, dvdy/12.0, dvdz/12.0);
	}
	
	def CalcElemVolumeDerivative(localCoords : Mat[_8, _3, Double]) : Mat[_3, _8, Double] = {
		val v0 = VoluDer(localCoords(_1,_0), localCoords(_2,_0), localCoords(_3,_0), localCoords(_4,_0), localCoords(_5,_0), localCoords(_7,_0), localCoords(_1,_1), localCoords(_2,_1), localCoords(_3,_1), localCoords(_4,_1), localCoords(_5,_1), localCoords(_7,_1), localCoords(_1,_2), localCoords(_2,_2), localCoords(_3,_2), localCoords(_4,_2), localCoords(_5,_2), localCoords(_7,_2));
		val v3 = VoluDer(localCoords(_0,_0), localCoords(_1,_0), localCoords(_2,_0), localCoords(_7,_0), localCoords(_4,_0), localCoords(_6,_0), localCoords(_0,_1), localCoords(_1,_1), localCoords(_2,_1), localCoords(_7,_1), localCoords(_4,_1), localCoords(_6,_1), localCoords(_0,_2), localCoords(_1,_2), localCoords(_2,_2), localCoords(_7,_2), localCoords(_4,_2), localCoords(_6,_2));
		val v2 = VoluDer(localCoords(_3,_0), localCoords(_0,_0), localCoords(_1,_0), localCoords(_6,_0), localCoords(_7,_0), localCoords(_5,_0), localCoords(_3,_1), localCoords(_0,_1), localCoords(_1,_1), localCoords(_6,_1), localCoords(_7,_1), localCoords(_5,_1), localCoords(_3,_2), localCoords(_0,_2), localCoords(_1,_2), localCoords(_6,_2), localCoords(_7,_2), localCoords(_5,_2));
		val v1 = VoluDer(localCoords(_2,_0), localCoords(_3,_0), localCoords(_0,_0), localCoords(_5,_0), localCoords(_6,_0), localCoords(_4,_0), localCoords(_2,_1), localCoords(_3,_1), localCoords(_0,_1), localCoords(_5,_1), localCoords(_6,_1), localCoords(_4,_1), localCoords(_2,_2), localCoords(_3,_2), localCoords(_0,_2), localCoords(_5,_2), localCoords(_6,_2), localCoords(_4,_2));
		val v4 = VoluDer(localCoords(_7,_0), localCoords(_6,_0), localCoords(_5,_0), localCoords(_0,_0), localCoords(_3,_0), localCoords(_1,_0), localCoords(_7,_1), localCoords(_6,_1), localCoords(_5,_1), localCoords(_0,_1), localCoords(_3,_1), localCoords(_1,_1), localCoords(_7,_2), localCoords(_6,_2), localCoords(_5,_2), localCoords(_0,_2), localCoords(_3,_2), localCoords(_1,_2));
		val v5 = VoluDer(localCoords(_4,_0), localCoords(_7,_0), localCoords(_6,_0), localCoords(_1,_0), localCoords(_0,_0), localCoords(_2,_0), localCoords(_4,_1), localCoords(_7,_1), localCoords(_6,_1), localCoords(_1,_1), localCoords(_0,_1), localCoords(_2,_1), localCoords(_4,_2), localCoords(_7,_2), localCoords(_6,_2), localCoords(_1,_2), localCoords(_0,_2), localCoords(_2,_2));
		val v6 = VoluDer(localCoords(_5,_0), localCoords(_4,_0), localCoords(_7,_0), localCoords(_2,_0), localCoords(_1,_0), localCoords(_3,_0), localCoords(_5,_1), localCoords(_4,_1), localCoords(_7,_1), localCoords(_2,_1), localCoords(_1,_1), localCoords(_3,_1), localCoords(_5,_2), localCoords(_4,_2), localCoords(_7,_2), localCoords(_2,_2), localCoords(_1,_2), localCoords(_3,_2));
		val v7 = VoluDer(localCoords(_6,_0), localCoords(_5,_0), localCoords(_4,_0), localCoords(_3,_0), localCoords(_2,_0), localCoords(_0,_0), localCoords(_6,_1), localCoords(_5,_1), localCoords(_4,_1), localCoords(_3,_1), localCoords(_2,_1), localCoords(_0,_1), localCoords(_6,_2), localCoords(_5,_2), localCoords(_4,_2), localCoords(_3,_2), localCoords(_2,_2), localCoords(_0,_2));
		Mat(Vec(v0.x, v1.x, v2.x, v3.x, v4.x, v5.x, v6.x, v7.x),
			Vec(v0.y, v1.y, v2.y, v3.y, v4.y, v5.y, v6.y, v7.y),
			Vec(v0.z, v1.z, v2.z, v3.z, v4.z, v5.z, v6.z, v7.z))
	}

	def CalcFBHourglassForceForElems() {
		for (c <- cells(mesh)) {
			val determ = volo(c) * v(c)
			val volinv = 1.0 / determ
			val ss1 = ss(c)
			val mass1 = elementMass(c)
			val volume13 = cbrt(determ)
			val coefficient = -hgcoef * 0.01 * ss1 * mass1 / volume13
   
			val localCoords = getLocalNodeCoordVectors(c)
			val pf = CalcElemVolumeDerivative(localCoords)
			val gamma = Mat(Vec( 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0), Vec( 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0),
							Vec( 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0), Vec(-1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0))
			val hourgam : Mat[_4, _8, Double] = gamma - (volinv * (gamma * localCoords * pf))
			val hourgamXpose : Mat[_8, _4, Double] = Mat(col(hourgam,0), col(hourgam,1), col(hourgam,2), col(hourgam,3), col(hourgam,4), col(hourgam,5), col(hourgam,6), col(hourgam,7));
			val localVelocities = getLocalNodeVelocityVectors(c);
			val hgf = coefficient * (hourgamXpose * (hourgam * localVelocities));

			val v0 = vertex(c,0)
			forces(v0) += row(hgf,0)

			val v1 = vertex(c,1)
			forces(v1) += row(hgf,1)
   
			val v2 = vertex(c,2)
			forces(v2) += row(hgf,2)

			val v3 = vertex(c,3)
			forces(v3) += row(hgf,3)

			val v4 = vertex(c,4)
			forces(v4) += row(hgf,4)

			val v5 = vertex(c,5)
			forces(v5) += row(hgf,5)

			val v6 = vertex(c,6)
			forces(v6) += row(hgf,6)
   
			val v7 = vertex(c,7)
			forces(v7) += row(hgf,7)
		}
	}

	def CalcVolumeForceForElems() {
		if (numElem > 0) {
			//val start_time = wall_time()
				IntegrateStressForElems();
			//val end_time = wall_time()
			//timeIntegrateStress += (end_time-start_time)
			if (hgcoef > 0.0) {
				//val start_time = wall_time()
					CalcFBHourglassForceForElems()
				//val end_time = wall_time()
				//timeHourglass += (end_time-start_time)
			}
		}
	}

	// TODO: Comment the next three functions for
	// implementing loop fusion of CalcPosition	
	/*def CalcAccelerationForNodes() {
		for (v <- vertices(mesh)) {
			val nMass = nodalMass(v)
			val ftmp = forces(v)
			xdd(v) = ftmp(_0)/nMass
			ydd(v) = ftmp(v)(_1)/nMass
			zdd(v) = ftmp(v)(_2)/nMass
		}
	}
	
	def ApplyAccelerationBoundaryConditionsForNodes() {
		for(v <- symmX) {
			xdd(v) = 0
		}
		for(v <- symmY) {
			ydd(v) = 0
		}
		for(v <- symmZ) {
			zdd(v) = 0
		}
	}

	def CalcPositionForNodes(dt : Double) {
		for (v <- vertices(mesh)) {
			val vtmp = velocity(v)
			var xtmp = vtmp(_0) + xdd(v)*dt
			var ytmp = vtmp(_1) + ydd(v)*dt
			var ztmp = vtmp(_2) + zdd(v)*dt
			val u_cut_tmp = u_cut
			xtmp = if(fabs(xtmp) < u_cut_tmp) 0.0 else xtmp
			ytmp = if(fabs(ytmp) < u_cut_tmp) 0.0 else ytmp
			ztmp = if(fabs(ztmp) < u_cut_tmp) 0.0 else ztmp
			velocity(v) = Vec(xtmp, ytmp, ztmp)
			position(v) += Vec(xtmp*dt, ytmp*dt, ztmp*dt)
			forces(v) = Vec(0.0,0.0,0.0)
		}
	}*/

	// TODO: Uncomment this function for
	// implementing loop fusion of CalcPosition	
	def CalcPositionForNodes(dt : Double) {
		for (v <- vertices(mesh)) {
			val nMass = nodalMass(v)
			var acceleration = forces(v)/nMass * Vec(symmX(v).toDouble, symmY(v).toDouble, symmZ(v).toDouble)
			val vtmp = velocity(v) + acceleration * dt
			val u_cut_tmp = u_cut
			val xtmp = if(fabs(vtmp(_0)) < u_cut_tmp) 0.0 else vtmp(_0)
			val ytmp = if(fabs(vtmp(_1)) < u_cut_tmp) 0.0 else vtmp(_1)
			val ztmp = if(fabs(vtmp(_2)) < u_cut_tmp) 0.0 else vtmp(_2)
			velocity(v) = Vec(xtmp, ytmp, ztmp)
			position(v) += Vec(xtmp*dt, ytmp*dt, ztmp*dt)
			forces(v) = Vec(0.0,0.0,0.0)
		}
	}
	
	def LagrangeNodal() {
		CalcVolumeForceForElems()
		//val start_time = wall_time()
			//TODO: Comment next two function for loop fusion of CalcPosition
			//CalcAccelerationForNodes()
			//ApplyAccelerationBoundaryConditionsForNodes()
			CalcPositionForNodes(deltatime)
		//val end_time = wall_time()
		//timePosition += (end_time-start_time)
	}


	// TODO: Order of the index method 
	// Get rid of order arithmetic on cell vertices
	// Cache optimizations/Reorder instructions?? 
	def CalcElemVolume(localCoords: Mat[_8, _3, Double]) : Double = {
		val d61 = row(localCoords,6) - row(localCoords,1)
		val d70 = row(localCoords,7) - row(localCoords,0)
		val d63 = row(localCoords,6) - row(localCoords,3)
		val d20 = row(localCoords,2) - row(localCoords,0)
		val d50 = row(localCoords,5) - row(localCoords,0)
		val d64 = row(localCoords,6) - row(localCoords,4)
		val d31 = row(localCoords,3) - row(localCoords,1)
		val d72 = row(localCoords,7) - row(localCoords,2)
		val d43 = row(localCoords,4) - row(localCoords,3)
		val d57 = row(localCoords,5) - row(localCoords,7)
		val d14 = row(localCoords,1) - row(localCoords,4)
		val d25 = row(localCoords,2) - row(localCoords,5)
		// val volume = TripleProduct(d31 + d72, d63, d20) + TripleProduct(d43 + d57, d64, d70) + TripleProduct(d14 + d25, d61, d50)
		val volume = dot(d31 + d72, cross(d63, d20)) + dot(d43 + d57, cross(d64, d70)) + dot(d14 + d25, cross(d61, d50))
		return volume / 12.0;
	}

	// Eliminated call to AreaFace(r1,r2,r3,r4)
	def CalcElemCharacteristicLength(localCoords : Mat[_8, _3, Double], volume : Double) : Double = {
		val r0 = row(localCoords, 0)
		val r1 = row(localCoords, 1)
		val r2 = row(localCoords, 2)
		val r3 = row(localCoords, 3)
		val r4 = row(localCoords, 4)
		val r5 = row(localCoords, 5)
		val r6 = row(localCoords, 6)
		val r7 = row(localCoords, 7)

		var f : Vec[_3, Double] = Vec(0.0, 0.0, 0.0)
		var g : Vec[_3, Double] = Vec(0.0, 0.0, 0.0)
		var temp : Double = 0.0
		var a : Double = 0.0
		var charlength : Double = 0.0

		f = (r2 - r0) - (r3 - r1)
		g = (r2 - r0) + (r3 - r1)
		temp = dot(f,g)
		a = dot(f,f)*dot(g,g) - temp*temp
		charlength = a max charlength;

		f = (r6 - r4) - (r7 - r5)
		g = (r6 - r4) + (r7 - r5)
		temp = dot(f,g)
		a = dot(f,f)*dot(g,g) - temp*temp
		charlength = a max charlength;

		f = (r5 - r0) - (r4 - r1)
		g = (r5 - r0) + (r4 - r1)
		temp = dot(f,g)
		a = dot(f,f)*dot(g,g) - temp*temp
		charlength = a max charlength;

		f = (r6 - r1) - (r5 - r2)
		g = (r6 - r1) + (r5 - r2)
		temp = dot(f,g)
		a = dot(f,f)*dot(g,g) - temp*temp
		charlength = a max charlength;

		f = (r7 - r2) - (r6 - r3)
		g = (r7 - r2) + (r6 - r3)
		temp = dot(f,g)
		a = dot(f,f)*dot(g,g) - temp*temp
		charlength = a max charlength;

		f = (r4 - r3) - (r7 - r0)
		g = (r4 - r3) + (r7 - r0)
		temp = dot(f,g)
		a = dot(f,f)*dot(g,g) - temp*temp
		charlength = a max charlength;

		charlength = 4.0 * volume / sqrt(charlength);

		return charlength;
	}

	//TODO: Pass by value semantics for vectors. So returning vector d
	def CalcElemVelocityGrandient(localVelocities : Mat[_8, _3, Double], pf : Mat[_4, _3, Double], detJ : Double) : Vec[_6, Double] = {

		val inv_detJ = 1.0 / detJ
		val r06 = row(localVelocities,0) - row(localVelocities,6)
		val r17 = row(localVelocities,1) - row(localVelocities,7)
		val r24 = row(localVelocities,2) - row(localVelocities,4)
		val r35 = row(localVelocities,3) - row(localVelocities,5)

		val pfr0 = row(pf,0)
		val pfr1 = row(pf,1)
		val pfr2 = row(pf,2)
		val pfr3 = row(pf,3)

		val temp = inv_detJ * ((pfr0 * r06) + (pfr1 * r17) + (pfr2 * r24) + (pfr3 * r35))

		//TODO: Improve this if possible
		val pfc0 = col(pf,0) // x-coords
		val pfc1 = col(pf,1) // y-coords
		val pfc2 = col(pf,2) // z-coords

		val dyddx = inv_detJ * (pfc0(_0) * r06(_1)
							+ pfc0(_1) * r17(_1)
							+ pfc0(_2) * r24(_1)
							+ pfc0(_3) * r35(_1));

		val dxddy = inv_detJ * (pfc1(_0) * r06(_0)
							+ pfc1(_1) * r17(_0)
							+ pfc1(_2) * r24(_0)
							+ pfc1(_3) * r35(_0));

		val dzddx = inv_detJ * (pfc0(_0) * r06(_2)
							+ pfc0(_1) * r17(_2)
							+ pfc0(_2) * r24(_2)
							+ pfc0(_3) * r35(_2));

		val dxddz = inv_detJ * (pfc2(_0) * r06(_0)
							+ pfc2(_1) * r17(_0)
							+ pfc2(_2) * r24(_0)
							+ pfc2(_3) * r35(_0));

		val dzddy = inv_detJ * (pfc1(_0) * r06(_2)
							+ pfc1(_1) * r17(_2)
							+ pfc1(_2) * r24(_2)
							+ pfc1(_3) * r35(_2));

		val dyddz = inv_detJ * (pfc2(_0) * r06(_1)
							+ pfc2(_1) * r17(_1)
							+ pfc2(_2) * r24(_1)
							+ pfc2(_3) * r35(_1));

		Vec(temp(_0), temp(_1), temp(_2), 0.5*(dzddy + dyddz), 0.5*(dxddz + dzddx), 0.5*(dxddy + dyddx))
	}


	def CalcKinematicsForElems(c : Cell, dxx: Field[Cell,Double], dyy: Field[Cell,Double], dzz: Field[Cell,Double], localCoords : Mat[_8,_3,Double], localVelocities : Mat[_8,_3,Double]) {
 		/* principal strains dxx, dyy, dzz */

		val volume = CalcElemVolume(localCoords)
		val relativeVolume = volume / volo(c)
		vnew(c) = relativeVolume
		if(relativeVolume <= 0.0) { exit(Constants.VolumeError) }
		delv(c) = relativeVolume - v(c)

		arealg(c) = CalcElemCharacteristicLength(localCoords, volume)

		val dt2 = 0.5 * deltatime
		var i = 0
		while (i < 8) {
			localCoords(i, 0) -= dt2 * localVelocities(i, 0)
			localCoords(i, 1) -= dt2 * localVelocities(i, 1)
			localCoords(i, 2) -= dt2 * localVelocities(i, 2)
			i+=1
		}

		// TODO: HACK HACK HACK!!!
		val b = CalcElemShapeFunctionDerivatives2(localCoords)
		val detJ = row(b,8).x

		val D = CalcElemVelocityGrandient(localVelocities, Mat(row(b,0), row(b,1), row(b,2), row(b,3)), detJ)
		val vdov_tmp  = D(_0) + D(_1) + D(_2)
		val vdovthird = vdov_tmp/3.0
		vdov(c) = vdov_tmp
		dxx(c) = D(_0) - vdovthird
		dyy(c) = D(_1) - vdovthird
		dzz(c) = D(_2) - vdovthird
	}

	def CalcMonotonicQGradientsForElems(c : Cell,
										delv_xi: Field[Cell,Double], delv_eta: Field[Cell,Double], delv_zeta: Field[Cell,Double],
										delx_xi: Field[Cell,Double], delx_eta: Field[Cell,Double], delx_zeta: Field[Cell,Double],
										localCoords : Mat[_8,_3,Double], localVelocities : Mat[_8,_3,Double]) {
 		/* velocity gradient delv_xi, delv_eta, delv_zeta*/
		/* position gradient delx_xi, delx_eta, delx_zeta*/

		val ptiny = 1.e-36

		// TODO: HACK HACK HACK!!!
		// TODO: Uncomment next 110 lines for MPI version
		// TODO: Comment next 110 lines for Serial/GPU versions
		val minIndex = minIdx(c) 

		var rc0 = Vec(0.0, 0.0, 0.0)
		var rc1 = Vec(0.0, 0.0, 0.0)
		var rc2 = Vec(0.0, 0.0, 0.0)
		var rc3 = Vec(0.0, 0.0, 0.0)
		var rc4 = Vec(0.0, 0.0, 0.0)
		var rc5 = Vec(0.0, 0.0, 0.0)
		var rc6 = Vec(0.0, 0.0, 0.0)
		var rc7 = Vec(0.0, 0.0, 0.0)

		var rv0 = Vec(0.0, 0.0, 0.0)
		var rv1 = Vec(0.0, 0.0, 0.0)
		var rv2 = Vec(0.0, 0.0, 0.0)
		var rv3 = Vec(0.0, 0.0, 0.0)
		var rv4 = Vec(0.0, 0.0, 0.0)
		var rv5 = Vec(0.0, 0.0, 0.0)
		var rv6 = Vec(0.0, 0.0, 0.0)
		var rv7 = Vec(0.0, 0.0, 0.0)

		val edgeNodes_tmp = edgeNodes

		val v0 = vertex(c,0)
		if      (ID(v0) == minIndex                                      ) { rc0 = row(localCoords,0); rv0 = row(localVelocities,0) }
		else if (ID(v0) == minIndex                                   + 1) { rc1 = row(localCoords,0); rv1 = row(localVelocities,0) }
		else if (ID(v0) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,0); rv2 = row(localVelocities,0) }
		else if (ID(v0) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,0); rv3 = row(localVelocities,0) }
		else if (ID(v0) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,0); rv4 = row(localVelocities,0) }
		else if (ID(v0) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,0); rv5 = row(localVelocities,0) }
		else if (ID(v0) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,0); rv6 = row(localVelocities,0) }
		else if (ID(v0) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,0); rv7 = row(localVelocities,0) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v0), "::", minIndex)

		val v1 = vertex(c,1)
		if      (ID(v1) == minIndex                                      ) { rc0 = row(localCoords,1); rv0 = row(localVelocities,1) }
		else if (ID(v1) == minIndex                                   + 1) { rc1 = row(localCoords,1); rv1 = row(localVelocities,1) }
		else if (ID(v1) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,1); rv2 = row(localVelocities,1) }
		else if (ID(v1) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,1); rv3 = row(localVelocities,1) }
		else if (ID(v1) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,1); rv4 = row(localVelocities,1) }
		else if (ID(v1) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,1); rv5 = row(localVelocities,1) }
		else if (ID(v1) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,1); rv6 = row(localVelocities,1) }
		else if (ID(v1) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,1); rv7 = row(localVelocities,1) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v1), "::", minIndex)

		val v2 = vertex(c,2)
		if      (ID(v2) == minIndex                                      ) { rc0 = row(localCoords,2); rv0 = row(localVelocities,2) }
		else if (ID(v2) == minIndex                                   + 1) { rc1 = row(localCoords,2); rv1 = row(localVelocities,2) }
		else if (ID(v2) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,2); rv2 = row(localVelocities,2) }
		else if (ID(v2) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,2); rv3 = row(localVelocities,2) }
		else if (ID(v2) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,2); rv4 = row(localVelocities,2) }
		else if (ID(v2) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,2); rv5 = row(localVelocities,2) }
		else if (ID(v2) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,2); rv6 = row(localVelocities,2) }
		else if (ID(v2) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,2); rv7 = row(localVelocities,2) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v2), "::", minIndex)
	
		val v3 = vertex(c,3)
		if      (ID(v3) == minIndex                                      ) { rc0 = row(localCoords,3); rv0 = row(localVelocities,3) }
		else if (ID(v3) == minIndex                                   + 1) { rc1 = row(localCoords,3); rv1 = row(localVelocities,3) }
		else if (ID(v3) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,3); rv2 = row(localVelocities,3) }
		else if (ID(v3) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,3); rv3 = row(localVelocities,3) }
		else if (ID(v3) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,3); rv4 = row(localVelocities,3) }
		else if (ID(v3) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,3); rv5 = row(localVelocities,3) }
		else if (ID(v3) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,3); rv6 = row(localVelocities,3) }
		else if (ID(v3) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,3); rv7 = row(localVelocities,3) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v3), "::", minIndex)
		
		val v4 = vertex(c,4)
		if      (ID(v4) == minIndex                                      ) { rc0 = row(localCoords,4); rv0 = row(localVelocities,4) }
		else if (ID(v4) == minIndex                                   + 1) { rc1 = row(localCoords,4); rv1 = row(localVelocities,4) }
		else if (ID(v4) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,4); rv2 = row(localVelocities,4) }
		else if (ID(v4) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,4); rv3 = row(localVelocities,4) }
		else if (ID(v4) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,4); rv4 = row(localVelocities,4) }
		else if (ID(v4) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,4); rv5 = row(localVelocities,4) }
		else if (ID(v4) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,4); rv6 = row(localVelocities,4) }
		else if (ID(v4) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,4); rv7 = row(localVelocities,4) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v4), "::", minIndex)

		val v5 = vertex(c,5)
		if      (ID(v5) == minIndex                                      ) { rc0 = row(localCoords,5); rv0 = row(localVelocities,5) }
		else if (ID(v5) == minIndex                                   + 1) { rc1 = row(localCoords,5); rv1 = row(localVelocities,5) }
		else if (ID(v5) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,5); rv2 = row(localVelocities,5) }
		else if (ID(v5) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,5); rv3 = row(localVelocities,5) }
		else if (ID(v5) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,5); rv4 = row(localVelocities,5) }
		else if (ID(v5) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,5); rv5 = row(localVelocities,5) }
		else if (ID(v5) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,5); rv6 = row(localVelocities,5) }
		else if (ID(v5) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,5); rv7 = row(localVelocities,5) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v5), "::", minIndex)
		
		val v6 = vertex(c,6)
		if      (ID(v6) == minIndex                                      ) { rc0 = row(localCoords,6); rv0 = row(localVelocities,6) }
		else if (ID(v6) == minIndex                                   + 1) { rc1 = row(localCoords,6); rv1 = row(localVelocities,6) }
		else if (ID(v6) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,6); rv2 = row(localVelocities,6) }
		else if (ID(v6) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,6); rv3 = row(localVelocities,6) }
		else if (ID(v6) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,6); rv4 = row(localVelocities,6) }
		else if (ID(v6) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,6); rv5 = row(localVelocities,6) }
		else if (ID(v6) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,6); rv6 = row(localVelocities,6) }
		else if (ID(v6) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,6); rv7 = row(localVelocities,6) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v6), "::", minIndex)
	
		val v7 = vertex(c,7)
		if      (ID(v7) == minIndex                                      ) { rc0 = row(localCoords,7); rv0 = row(localVelocities,7) }
		else if (ID(v7) == minIndex                                   + 1) { rc1 = row(localCoords,7); rv1 = row(localVelocities,7) }
		else if (ID(v7) == minIndex                       + edgeNodes_tmp + 1) { rc2 = row(localCoords,7); rv2 = row(localVelocities,7) }
		else if (ID(v7) == minIndex                       + edgeNodes_tmp    ) { rc3 = row(localCoords,7); rv3 = row(localVelocities,7) }
		else if (ID(v7) == minIndex + edgeNodes_tmp*edgeNodes_tmp                ) { rc4 = row(localCoords,7); rv4 = row(localVelocities,7) }
		else if (ID(v7) == minIndex + edgeNodes_tmp*edgeNodes_tmp             + 1) { rc5 = row(localCoords,7); rv5 = row(localVelocities,7) }
		else if (ID(v7) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp + 1) { rc6 = row(localCoords,7); rv6 = row(localVelocities,7) }
		else if (ID(v7) == minIndex + edgeNodes_tmp*edgeNodes_tmp + edgeNodes_tmp    ) { rc7 = row(localCoords,7); rv7 = row(localVelocities,7) }
		else Print("Error: Cannot assign local index to vertex: ", ID(c), "::", ID(v7), "::", minIndex)


		// TODO: Uncomment next 8 lines for GPU/Serial vesions
		// TODO: Comment next 8 lines for MPI version
		//val rc0 = row(localCoords,7); val rv0 = row(localVelocities,7)
		//val rc1 = row(localCoords,6); val rv1 = row(localVelocities,6)
		//val rc2 = row(localCoords,5); val rv2 = row(localVelocities,5)
		//val rc3 = row(localCoords,4); val rv3 = row(localVelocities,4)
		//val rc4 = row(localCoords,3); val rv4 = row(localVelocities,3)
		//val rc5 = row(localCoords,2); val rv5 = row(localVelocities,2)
		//val rc6 = row(localCoords,1); val rv6 = row(localVelocities,1)
		//val rc7 = row(localCoords,0); val rv7 = row(localVelocities,0)


		val vol = volo(c)*vnew(c) 
		val norm = 1.0 / (vol + ptiny) 

		val dj = -0.25 * ((rc0 + rc1 + rc5 + rc4) - (rc3 + rc2 + rc6 + rc7));
		val di =  0.25 * ((rc1 + rc2 + rc6 + rc5) - (rc0 + rc3 + rc7 + rc4));
		val dk =  0.25 * ((rc4 + rc5 + rc6 + rc7) - (rc0 + rc1 + rc2 + rc3));

		/* find delvk and delxk ( i cross j ) */
		var a = cross(di,dj)
		delx_zeta(c) = vol / sqrt(dot(a,a) + ptiny) ;
		a *= norm;
		var dv = 0.25 * ((rv4 + rv5 + rv6 + rv7) - (rv0 + rv1 + rv2 + rv3));
		delv_zeta(c) = dot(a, dv);

		/* find delxi and delvi ( j cross k ) */
		a = cross(dj,dk)
		delx_xi(c) = vol / sqrt(dot(a,a) + ptiny) ;
		a *= norm;
		dv = 0.25 * ((rv1 + rv2 + rv6 + rv5) - (rv0 + rv3 + rv7 + rv4)) ;
		delv_xi(c) = dot(a, dv);

		/* find delxj and delvj ( k cross i ) */
		a = cross(dk,di)
		delx_eta(c) = vol / sqrt(dot(a,a) + ptiny) ;
		a *= norm;
		dv = -0.25 * ((rv0 + rv1 + rv5 + rv4) - (rv3 + rv2 + rv6 + rv7)) ;
		delv_eta(c) = dot(a, dv);
	}

	// TODO: IMPORTANT!!! Add abstractions in Liszt to:
	// 1. Support arbitrary (ordered) sets of mesh elements
	// 2. Read such arbitrary sets from input mesh file
	def CalcMonotonicQRegionForElems(c : Cell) {
 		/* velocity gradient */
		val delv_xi = scratchpade01
		val delv_eta = scratchpade02
		val delv_zeta = scratchpade03

		/* position gradient */
		val delx_xi = scratchpade04
		val delx_eta = scratchpade05
		val delx_zeta = scratchpade06

		val ptiny = 1.e-36
	
		var qlin : Double = 0.0
		var qquad : Double = 0.0
		var phixi : Double = 0.0
		var phieta : Double = 0.0
		var phizeta : Double = 0.0

		var norm : Double = 0.0

		val nc = neighborIndexes(c)
		val lxim   = nc(_0)
		val lxip   = nc(_1)
		val letam  = nc(_2)
		val letap  = nc(_3)
		val lzetam = nc(_4)
		val lzetap = nc(_5)

		var delvm : Double = 0.0
		var delvp : Double = 0.0

		val delv_xi_tmp   = delv_xi(c)
		val delv_eta_tmp  = delv_eta(c)
		val delv_zeta_tmp = delv_zeta(c)

		val monoq_limiter_mult_tmp = monoq_limiter_mult
		val monoq_max_slope_tmp = monoq_max_slope

		/* phixi */
		norm = 1.0 / (delv_xi_tmp + ptiny)

		// TODO: Change these for specific support for hexahedrons whenever available
		if(lxim == -1) { // On X-symmetry plane
			delvm = delv_xi_tmp
		}
		else {
			delvm = delv_xi(cell(c, lxim))
		}
		if(lxip == -1) { // On X-free plane
			delvp = 0.0 
		}
		else {
			delvp = delv_xi(cell(c, lxip))
		}

		delvm = delvm * norm
		delvp = delvp * norm

		phixi = 0.5 * (delvm + delvp)
		delvm *= monoq_limiter_mult_tmp
		delvp *= monoq_limiter_mult_tmp

		if(delvm < phixi) phixi = delvm
		if(delvp < phixi) phixi = delvp
		if(phixi < 0.0) phixi = 0.0
		if(phixi > monoq_max_slope_tmp) phixi = monoq_max_slope_tmp

		/* phieta */
		norm = 1.0 / (delv_eta_tmp + ptiny)

		if(letam == -1) { // On Y-symmetry plane
			delvm = delv_eta_tmp
		}
		else {
			delvm = delv_eta(cell(c, letam))
		}

		if(letap == -1) { // On Y-free plane
			delvp = 0.0 
		}
		else {
			delvp = delv_eta(cell(c, letap))
		}
	
		delvm = delvm * norm 
		delvp = delvp * norm 

		phieta = 0.5 * (delvm + delvp) 
		delvm *= monoq_limiter_mult_tmp
		delvp *= monoq_limiter_mult_tmp

		if(delvm  < phieta ) phieta = delvm 
		if(delvp  < phieta ) phieta = delvp 
		if(phieta < 0.0) phieta = 0.0 
		if(phieta > monoq_max_slope_tmp) phieta = monoq_max_slope_tmp

		/* phizeta */
		norm = 1.0 / (delv_zeta_tmp + ptiny)

		if(lzetam == -1) { // On Z-symmetry plane
			delvm = delv_zeta_tmp
		}
		else {
			delvm = delv_zeta(cell(c, lzetam))
		}

		if(lzetap == -1) { // On Z-free plane
			delvp = 0.0 
		}
		else {
			delvp = delv_zeta(cell(c, lzetap))
		}

		delvm = delvm * norm 
		delvp = delvp * norm 

		phizeta = 0.5 * (delvm + delvp) 
		delvm *= monoq_limiter_mult_tmp
		delvp *= monoq_limiter_mult_tmp

		if(delvm  < phizeta ) phizeta = delvm 
		if(delvp  < phizeta ) phizeta = delvp 
		if(phizeta < 0.0) phizeta = 0.0 
		if(phizeta > monoq_max_slope_tmp) phizeta = monoq_max_slope_tmp

		/* Remove length scale */
		if (vdov(c) > 0.0) {
			qlin  = 0.0
			qquad = 0.0
		}
		else {
			var delvxxi   = delv_xi_tmp   * delx_xi(c)
			var delvxeta  = delv_eta_tmp  * delx_eta(c)
			var delvxzeta = delv_zeta_tmp * delx_zeta(c)
			if(delvxxi   > 0.0) delvxxi   = 0.0
			if(delvxeta  > 0.0) delvxeta  = 0.0
			if(delvxzeta > 0.0) delvxzeta = 0.0

			val rho = elementMass(c) / (volo(c) * vnew(c))
			qlin = -qlc_monoq * rho * (delvxxi * (1.0 - phixi) + delvxeta * (1.0 - phieta) + delvxzeta * (1.0 - phizeta))
			qquad = qqc_monoq * rho * (delvxxi*delvxxi * (1.0 - phixi*phixi) + delvxeta*delvxeta * (1.0 - phieta*phieta) + delvxzeta*delvxzeta * (1.0 - phizeta*phizeta))
		}

		qq(c) = qquad
		ql(c) = qlin

		if ( qlin > qstop ) {
				exit(Constants.QStopError) 
		}
	}

	def CalcPressureForElems(e_old : Double, compression: Double, vnewc : Double,
							 pmin : Double, p_cut : Double, eosvmax : Double) : Vec[_3, Double] = {
		val c1s = 2.0/3.0
		val bvc = c1s * (compression + 1.0)
		val	pbvc = c1s
		var p_new = bvc * e_old

		if(fabs(p_new) < p_cut) {
			p_new = 0.0
		}
		if(vnewc >= eosvmax) {
			p_new = 0.0
		}
		p_new = p_new max pmin

		Vec(p_new, bvc, pbvc)
	}

	def CalcEnergyForElems(p_old : Double, e_old : Double, q_old : Double,
						   compression : Double, compHalfStep : Double, vnewc : Double, work : Double,
						   delvc : Double, qq_old : Double, ql_old : Double, eosvmax : Double, rho0 : Double) : Vec[_5, Double] = {
		val sixth = 1.0/6.0

		val emin_tmp = emin

		var e_new = (e_old - 0.5 * delvc * (p_old + q_old) + 0.5 * work) max emin_tmp

		val p_cut_tmp = p_cut
		val pmin_tmp = pmin

		var retVal = CalcPressureForElems(e_new, compHalfStep, vnewc, pmin_tmp, p_cut_tmp, eosvmax)
		val pHalfStep = retVal(_0)
		var bvc       = retVal(_1)
		var pbvc      = retVal(_2)
		val vhalf = 1.0 / (1.0 + compHalfStep)
		var ssc = CalcSoundSpeedForElem(vhalf, rho0, e_new, pHalfStep, pbvc, bvc);
		var q_new = 0.0
		if(delvc <= 0.0) q_new = (ssc*ql_old + qq_old)
		e_new = e_new + 0.5 * delvc * (3.0 * (p_old + q_old) - 4.0 * (pHalfStep + q_new)) + (0.5 * work)

		val e_cut_tmp = e_cut

		if(fabs(e_new) < e_cut_tmp) e_new = 0.0
		e_new = e_new max emin_tmp


		retVal = CalcPressureForElems(e_new, compression, vnewc, pmin_tmp, p_cut_tmp, eosvmax)
		var p_new     = retVal(_0)
		bvc       = retVal(_1)
		pbvc      = retVal(_2)
		ssc = CalcSoundSpeedForElem(vnewc, rho0, e_new, p_new, pbvc, bvc);
		var q_tilde = 0.0
		if(delvc <= 0.0) q_tilde = (ssc*ql_old + qq_old)
		e_new = e_new - (7.0*(p_old + q_old) - 8.0*(pHalfStep + q_new) + (p_new + q_tilde)) * delvc * sixth
		if(fabs(e_new) < e_cut_tmp) e_new = 0.0;
		e_new = e_new max emin_tmp


		retVal = CalcPressureForElems(e_new, compression, vnewc, pmin_tmp, p_cut_tmp, eosvmax)
		p_new     = retVal(_0)
		bvc       = retVal(_1)
		pbvc      = retVal(_2)
		if(delvc <= 0.0) {
			ssc = CalcSoundSpeedForElem(vnewc, rho0, e_new, p_new, pbvc, bvc);
			q_new = (ssc*ql_old + qq_old)
			if (fabs(q_new) < q_cut) {
				q_new = 0.0 
			}
		}

		Vec(p_new, e_new, q_new, bvc, pbvc)
	}

	def CalcSoundSpeedForElem(vnewc : Double, rho0 : Double, enewc : Double, pnewc : Double, pbvc : Double, bvc : Double) : Double = {
		var ss = (pbvc * enewc + vnewc * vnewc * bvc * pnewc) / rho0
		if (ss <= 0.111111e-36) {
			ss = 0.3333333e-18
		}
		else {
			ss = sqrt(ss)
		}
		return ss
	}

	def EvalEOSForElems(c : Cell) {
		val rho0         = refdens
		val vnewc        = vnew(c);
		val delvc        = delv(c);
		val e_old        = e(c);
		var p_old        = p(c);
		val q_old        = q(c);
		val qqtmp        = qq(c);
		val qltmp        = ql(c);
		val work         = 0.0 ;
		var compression  = 1.0 / vnewc - 1.0
		val vchalf       = vnewc - delvc * 0.5
		var compHalfStep = 1.0 / vchalf - 1.0

		val eosvmax_tmp = eosvmax
		val eosvmin_tmp = eosvmin

		/* Check for v > eosvmax or v < eosvmin */
		if((eosvmin_tmp != 0.0) && (vnewc < eosvmin_tmp)) {
			compHalfStep = compression
		}

		if((eosvmax_tmp != 0.0) && (vnewc > eosvmax_tmp)) {
			p_old        = 0.0
			compression  = 0.0
			compHalfStep = 0.0
		}

		val peqbvpb = CalcEnergyForElems(p_old, e_old, q_old, compression, compHalfStep, vnewc, work, delvc, qqtmp, qltmp, eosvmax_tmp, rho0)

		val p_new = peqbvpb(_0)
		val e_new = peqbvpb(_1)
		val q_new = peqbvpb(_2)
		val bvc   = peqbvpb(_3)
		val pbvc  = peqbvpb(_4)

		val ssc = CalcSoundSpeedForElem(vnewc, rho0, e_new, p_new, pbvc, bvc)

		p(c) = p_new
		e(c) = e_new
		q(c) = q_new
		ss(c) = ssc
	}

	def UpdateVolumesForElems(c : Cell) {
			var tmpV = vnew(c) 
			if (fabs(tmpV - 1.0) < v_cut) tmpV = 1.0 
			v(c) = tmpV
	}

	//Eliminated call to CalcLagrangeElements: Fused CalcKinematicsForElems and calculation of vdovthird
	def LagrangeElements() {
		//val start_time1 = wall_time()
			for(c <- cells(mesh)) {
				// TODO: Implement pass-by-reference semantics
				val localCoords = getLocalNodeCoordVectors(c)
				val localVelocities = getLocalNodeVelocityVectors(c)
				CalcKinematicsForElems(c, scratchpade01, scratchpade02, scratchpade03, localCoords, localVelocities) 
				CalcMonotonicQGradientsForElems(c, scratchpade01, scratchpade02, scratchpade03, scratchpade04, scratchpade05, scratchpade06, localCoords, localVelocities)
			}
		//val end_time1 = wall_time()
		//timeKinQ += (end_time1-start_time1)

		//val start_time2 = wall_time()
			for(c <- material) {
				CalcMonotonicQRegionForElems(c)
				EvalEOSForElems(c)
			}
		//val end_time2 = wall_time()
		//timeQRegionEOS += (end_time2-start_time2)

		//val start_time3 = wall_time()
			for(c <- cells(mesh)) {
				UpdateVolumesForElems(c) 
			}
		//val end_time3 = wall_time()
		//timeUpdateVol += (end_time3-start_time3)
	}

	// TODO: Add minloc reduction to Liszt
	def CalcCourantConstraintForElems() {
		var dtcourant_tmp = 1.0e+20

		val qqc_tmp = qqc

		var qqc2 = 64.0 * qqc_tmp * qqc_tmp

		for (c <- material) {
			val ssc = ss(c)
			val vdovtmp = vdov(c)
			val arealgtmp = arealg(c)

			var dtf = ssc * ssc
      		if(vdovtmp < 0.0) {
				dtf += qqc2 * arealgtmp * arealgtmp * vdovtmp * vdovtmp
			}
			dtf = sqrt(dtf)
			dtf = arealgtmp / dtf
			if (vdovtmp != 0.0) {
				dtcourant_tmp = dtcourant_tmp min dtf
			}
		}

		if (dtcourant_tmp != 1.0e+20) {
			dtcourant = dtcourant_tmp 
		}
	}

	def CalcHydroConstraintForElems() {
		var dthydro_tmp = 1.0e+20 

		for (c <- material) {
			val vdovtmp = vdov(c)
			if(vdovtmp != 0.0) {
				val dtdvov = dvovmax / (fabs(vdovtmp) + 1.e-20)
				dthydro_tmp = dthydro_tmp min dtdvov
			}
		}

		if (dthydro_tmp != 1.0e+20) {
			dthydro = dthydro_tmp 
		}
	}

	def CalcTimeConstraintsForElems() {
		/* evaluate time constraint */
		//val start_time1 = wall_time()
			CalcCourantConstraintForElems() 
		//val end_time1 = wall_time()
		//timeCourant += (end_time1-start_time1)

		/* check hydro constraint */
		//val start_time2 = wall_time()
			CalcHydroConstraintForElems() 
		//val end_time2 = wall_time()
		//timeHydro+= (end_time2-start_time2)
	}

	def LagrangeLeapFrog() {
		/* calculate nodal forces, accelerations, velocities, positions, with
		* applied boundary conditions and slide surface considerations */
		LagrangeNodal()

		/* calculate element quantities (i.e. velocity gradient & q), and update
		* material states */
		LagrangeElements()

		CalcTimeConstraintsForElems()
	}

	def runSolver() {
		/*while(cycle < 200) {
			TimeIncrement()
			LagrangeLeapFrog()
			if(Constants.DEBUG_FLAG) {
				Print("cycle: ", cycle, ", time = ", time, ", dt = ", deltatime)
			}
			//cycle += 1
		}*/

		while(time < stoptime) {
			TimeIncrement()
			LagrangeLeapFrog()
			//if(Constants.DEBUG_FLAG) {
			//	Print("time = ", time, ", dt = ", deltatime)
			//}
		}
	}

	def stats() {
		Print("Total elapsed time = ", (end_time - start_time))
		//Print("Courant time       = ", timeCourant)
		//Print("Hydro time         = ", timeHydro)
		//Print("Position time      = ", timePosition)
		//Print("Vol Update time    = ", timeUpdateVol)
		//Print("Int Stress time    = ", timeIntegrateStress)
		//Print("Hourglass Time     = ", timeHourglass)
		//Print("Kine + MonoQGrad   = ", timeKinQ)
		//Print("QReg + EOS         = ", timeQRegionEOS)
		//Print("Total time         = ", (timeCourant + timeHydro + timePosition + timeUpdateVol + timeIntegrateStress + timeHourglass + timeKinQ + timeQRegionEOS))
		//Print("Run completed:")
		Print("   Problem size        =  ", edgeElems)
		Print("   Iteration count     =  ", cycle)
		// TODO: add origin node
		var finalEnergy = 0.0
		for(c <- cells(mesh)) if(ID(c) == 1) finalEnergy = e(c)
		Print("   Final Origin Energy =  ", finalEnergy)
	}


	/*********************************************************************************************/
	/* HELPER FUNCTIONS */
	/*********************************************************************************************/
	// TODO: Add System.exit in Liszt
	def exit(errorCode : Int) {
		Print("Exiting: ", errorCode)
		//System.exit(errorCode)
	}

	// Cannot be done because in Liszt, functions cannot return Fields :(
	/*def getPad() : Field[Cell, Double] = {
		val pad : Field[Cell, Double] =
			 if(padNum==0) { scratchpade01 }
		else if(padNum==1) { scratchpade02 }
		else if(padNum==2) { scratchpade03 }
		else if(padNum==3) { scratchpade04 }
		else if(padNum==4) { scratchpade05 }
		else { scratchpade06 }
		padNum = (padNum+1)%6
		pad
	}*/

	def getLocalNodeCoordVectors(c : Cell) : Mat[_8, _3, Double] = {
		val v0 = vertex(c,0)
		val v1 = vertex(c,1)
		val v2 = vertex(c,2)
		val v3 = vertex(c,3)
		val v4 = vertex(c,4)
		val v5 = vertex(c,5)
		val v6 = vertex(c,6)
		val v7 = vertex(c,7)
	
		Mat(position(v0), position(v1), position(v2), position(v3), position(v4), position(v5), position(v6), position(v7))
	}

	def getLocalNodeVelocityVectors(c : Cell) : Mat[_8, _3, Double] = {
		val v0 = vertex(c,0)
		val v1 = vertex(c,1)
		val v2 = vertex(c,2)
		val v3 = vertex(c,3)
		val v4 = vertex(c,4)
		val v5 = vertex(c,5)
		val v6 = vertex(c,6)
		val v7 = vertex(c,7)

		Mat(velocity(v0), velocity(v1), velocity(v2), velocity(v3), velocity(v4), velocity(v5), velocity(v6), velocity(v7))
	}

	def BuildMeshOrientation() {
		SetLocalNeighborIndexVectors()
	}

	//TODO: Set these to be read from input mesh file
	def SetLocalNeighborIndexVectors() {
		for (c <- cells(mesh)) {
			var lxim   = -1
			var lxip   = -1
			var letam  = -1
			var letap  = -1
			var lzetam = -1
			var lzetap = -1
	
			val cId = ID(c) // Ids are from 1..numElem

			var ncId = 0
			val xp = cId + 1
			val xm = cId - 1
			val yp = cId + edgeElems
			val ym = cId - edgeElems
			val zp = cId + edgeElems*edgeElems
			val zm = cId - edgeElems*edgeElems
	

			val c0 = cell(c, 0)
			ncId = ID(c0)
			if(ncId != 0) {
				if      (ncId == xp) lxip   = 0
				else if (ncId == xm) lxim   = 0
				else if (ncId == yp) letap  = 0
				else if (ncId == ym) letam  = 0
				else if (ncId == zp) lzetap = 0
				else if (ncId == zm) lzetam = 0
				else Print("Error: Cannot assign local index to neighbor cell: ", cId, "::", ncId)
			}

			val c1 = cell(c, 1)
			ncId = ID(c1)
			if(ncId != 0) {
				if      (ncId == xp) lxip   = 1
				else if (ncId == xm) lxim   = 1
				else if (ncId == yp) letap  = 1
				else if (ncId == ym) letam  = 1
				else if (ncId == zp) lzetap = 1
				else if (ncId == zm) lzetam = 1
				else Print("Error: Cannot assign local index to neighbor cell: ", cId, "::", ncId)
			}

			val c2 = cell(c, 2)
			ncId = ID(c2)
			if(ncId != 0) {
				if      (ncId == xp) lxip   = 2
				else if (ncId == xm) lxim   = 2
				else if (ncId == yp) letap  = 2
				else if (ncId == ym) letam  = 2
				else if (ncId == zp) lzetap = 2
				else if (ncId == zm) lzetam = 2
				else Print("Error: Cannot assign local index to neighbor cell: ", cId, "::", ncId)
			}

			val c3 = cell(c, 3)
			ncId = ID(c3)
			if(ncId != 0) {
				if      (ncId == xp) lxip   = 3
				else if (ncId == xm) lxim   = 3
				else if (ncId == yp) letap  = 3
				else if (ncId == ym) letam  = 3
				else if (ncId == zp) lzetap = 3
				else if (ncId == zm) lzetam = 3
				else Print("Error: Cannot assign local index to neighbor cell: ", cId, "::", ncId)
			}

			val c4 = cell(c, 4)
			ncId = ID(c4)
			if(ncId != 0) {
				if      (ncId == xp) lxip   = 4
				else if (ncId == xm) lxim   = 4
				else if (ncId == yp) letap  = 4
				else if (ncId == ym) letam  = 4
				else if (ncId == zp) lzetap = 4
				else if (ncId == zm) lzetam = 4
				else Print("Error: Cannot assign local index to neighbor cell: ", cId, "::", ncId)
			}

			val c5 = cell(c, 5)
			ncId = ID(c5)
			if(ncId != 0) {
				if      (ncId == xp) lxip   = 5
				else if (ncId == xm) lxim   = 5
				else if (ncId == yp) letap  = 5
				else if (ncId == ym) letam  = 5
				else if (ncId == zp) lzetap = 5
				else if (ncId == zm) lzetam = 5
				else Print("Error: Cannot assign local index to neighbor cell: ", cId, "::", ncId)
			}
			neighborIndexes(c) = Vec(lxim, lxip, letam, letap, lzetam, lzetap)
		}
	}
}

@lisztcode
object Lulesh {
	def main() {	
		Mesh.initMeshParameters()
		Mesh.start_time = wall_time()
		Mesh.runSolver()
		Mesh.end_time = wall_time()
		Mesh.stats()
	}
}