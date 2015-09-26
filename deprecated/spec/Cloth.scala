import Liszt.Language._
import Liszt.MetaInteger._


@lisztcode 
trait VecVertexLS[MO <: MeshObj, N <: IntM] extends LinearSystem {
	def row(v : MO) : Vec[N,RowIndex] = AutoIndex
	def col(v : MO) : Vec[N,ColIndex] = AutoIndex
}

@lisztcode
object MyLS extends VecVertexLS[Vertex,_3] with SuperLU

@lisztcode
object Main {
	
	val Position = FieldWithLabel[Vertex,Vec[_3,Double]]("position")
	
	val Velocity = FieldWithConst[Vertex,Vec[_3,Double]](Vec(0.,0.,0.))
	val Force= FieldWithConst[Vertex,Vec[_3,Double]](Vec(0.,0.,0.))
	
	val Mass = FieldWithConst[Vertex,Double]( 1.f/ size(vertices(mesh)) ) 
	
	val RestLength = FieldWithConst[Edge,Double]( 0. )
	
	val constrained = BoundarySet[Vertex]("constrained")
	
	val IsConstrained = FieldWithConst[Vertex,Int]( 0 )
	
	
	val interior = BoundarySet[Vertex]("interior")
	
	
	val h = .03
	val ks = 100.0
	val kd = 5.0
	val totaltime = 100.0
	
	def vdb_color(xx : Double, yy : Double, zz : Double) : Unit = __
	def vdb_frame() : Unit = __
	def vdb_begin() : Unit = __
	def vdb_end() : Unit = __
	def vdb_point(xx : Double, yy : Double, zz : Double) : Unit = __
	def vdb_line(xx : Double, yy : Double, zz : Double,x : Double, y : Double, z : Double) : Unit = __
	
	def point(x : Vec[_3,Double]) {
		vdb_point(x.x,x.y,x.z)
	}
	def line(x : Vec[_3,Double], y : Vec[_3,Double]) {
		vdb_line(x.x,x.y,x.z,y.x,y.y,y.z)
	}
	
	def print() {
		vdb_begin()
		vdb_frame()
		vdb_color(0,0,1.)
		vdb_line(0.,0.,0.,0.,0.,1)
		vdb_color(.5,.5,.5)
		for(e <- edges(mesh)) {
			line(Position(head(e)),Position(tail(e)))
		}
		val nV : Double = size(vertices(mesh))
		for(v <- interior) {
			vdb_color(1. ,0.,0.)
			point(Position(v))
		}
		vdb_color(0.,1.,0.)
		for(v <- constrained) {
			point(Position(v))
		}
		vdb_end()
	}
	
	def force_spring(e : Edge) : Vec[_3,Double] = {
		val pa = Position(head(e))
		val pb = Position(tail(e))
		val va = Velocity(head(e))
		val vb = Velocity(tail(e))
		val rl = RestLength(e)
		val l = length(pa-pb)
		-ks *(pa-pb) / l * (l - rl);
	}
	def force_damp(e : Edge) : Vec[_3,Double] = {
		val pa = Position(head(e))
		val pb = Position(tail(e))
		val va = Velocity(head(e))
		val vb = Velocity(tail(e))
		val diff = pa - pb
		-kd * dot(diff, va - vb) / dot(diff,diff) * diff
	}
	
	def jdap(e : Edge) : Mat[_3,_3,Double] = {
		val dX = Position(head(e)) - Position(tail(e))
		val dV = Velocity(head(e)) - Velocity(tail(e))
		val outerPP = outer(dX,dX)
		val outerPV = outer(dV,dV)
		val dotPP = dot(dX,dX)
		val dotPV = dot(dV,dV)
		
		val a = outerPV / dotPP
		val b = 2*((-1*dotPV)/(dotPP*dotPP))*outerPP;
		val c = dotPV / dotPP * Mat(Vec(1.,0.,0.),
		                            Vec(0.,1.,0.),
		                            Vec(0.,0.,1.))
		-kd * (a + b + c)
	}
	def jdav(e : Edge) : Mat[_3,_3,Double] = {
		val dX = Position(head(e)) - Position(tail(e))
		val outerPP = outer(dX,dX)
		val dotPP = dot(dX,dX)
		-kd * outerPP / dotPP
	}
	
	def jsap(e : Edge) : Mat[_3,_3,Double] = {
		val dX = Position(head(e)) - Position(tail(e))
		val outerP = outer(dX,dX)
		val lenSquared = dot(dX,dX)
		val len = sqrt(lenSquared)
		val I = Mat(Vec(1.,0.,0.),
		            Vec(0.,1.,0.),
		            Vec(0.,0.,1.))
		val a = (1 - RestLength(e)/len) * (I - outerP/lenSquared)
		val b = outerP / lenSquared
		-ks * ( a + b)
	}
	
	val A = MyLS.A()
	val delta_v = MyLS.x()
	val rhs = MyLS.b()
	val op_rhs = FieldWithConst[Vertex,Vec[_3,Double]](Vec(0.,0.,0.))
		
		
		
	//what the operator might look for constructing the sparse matrix in this example
	//this code hasn't been verified to produce the same matrix as the code that explicitly
	//constructs the matrix below
	def operator() {
	
		for(v <- vertices(mesh)) {
			val col = MyLS.col(v)
			op_rhs(v) =  Mass(v) * delta_v(col)
		}
		for(e <- edges(mesh)) {
			val JP_fa_xa = h*h*(jdap(e) + jsap(e))
			val JV_fa_xa = h*jdav(e)
			val ca = MyLS.col(head(e))
			val cb = MyLS.col(tail(e))
			
			val J = JP_fa_xa + JV_fa_xa
			
			op_rhs(head(e)) -= J * delta_v(ca)
			op_rhs(head(e)) += J * delta_v(cb)
				
				
			op_rhs(tail(e)) += J * delta_v(ca)
			op_rhs(tail(e)) -= J * delta_v(cb)
		}
		
	}
	
	def main() {
		
		MyLS.nonzeroes {
			nz =>
			for(v <- vertices(mesh)) {
				val r = MyLS.row(v)
				val c = MyLS.col(v)
				nz.b(r)
				nz.x(c)
				nz.A(r,c)
				if(size(vertices(v)) >= 6) {
					for(v2  <- vertices(v)) {
						nz.A(r,MyLS.col(v2))
					}
				}
			}
		}
		
		
		for(e <- edges(mesh)) {
			RestLength(e) = length(Position(head(e)) - Position(tail(e)))
		}
		for(v <- constrained) {
			IsConstrained(v) = 1
		}
		
		print()
		var t = 0.0
		val start = wall_time()
		var next_d = 1.f / 60.f
		while(t < totaltime) {
			
			for(v <- interior) {
				Force(v) = Vec(0.,0.,0.)
			}
			for(e <- edges(mesh)) {
				Force(head(e)) += force_spring(e) + force_damp(e)
				Force(tail(e)) += force_spring(flip(e)) + force_damp(flip(e))
			}
			for(v <- vertices(mesh)) {
				Force(v) += Mass(v) * Vec(0.,0.,9.8) // gravity
			}
			
			
			
			//explicit timestep
			/*
			for(v <- interior) {
				Position(v) += h * Velocity(v)
			}
			for(v <- interior) {
				Velocity(v) += h * Force(v)  / Mass(v)
			}
			for(v <- constrained) {
				Velocity(v) = Vec(0.,0.,0.)
			}*/
			
			//implicit timestep
			
			
			for(v <- vertices(mesh)) {
				val row = MyLS.row(v)
				val col = MyLS.col(v)
				val I = Mat(Vec(1.,0.,0.),
				            Vec(0.,1.,0.),
				            Vec(0.,0.,1.))
				A(row,col) = Mass(v) * I
				rhs(row) = h * Force(v)
			}
			for(e <- edges(mesh)) {
				val JP_fa_xa = h*h*(jdap(e) + jsap(e))
				val JV_fa_xa = h*jdav(e)
				val ra = MyLS.row(head(e))
				val ca = MyLS.col(head(e))
				val rb = MyLS.row(tail(e))
				val cb = MyLS.col(tail(e))
				
				val J = JP_fa_xa + JV_fa_xa
				A(ra,ca) -= J
				A(ra,cb) += J
				rhs(ra) += JP_fa_xa * Velocity(head(e))
				rhs(ra) -= JP_fa_xa * Velocity(tail(e))
				
				
				A(rb,ca) += J
				A(rb,cb) -= J
				rhs(rb) -= JP_fa_xa * Velocity(head(e))
				rhs(rb) += JP_fa_xa * Velocity(tail(e))
			}
			
			//zero out the constraints
			for(v <- constrained) {
				val rv = MyLS.row(v)
				val cv = MyLS.col(v)
				val Z = Mat(Vec(0.,0.,0.),
				            Vec(0.,0.,0.),
				            Vec(0.,0.,0.))
				val vZ = Vec(0.,0.,0.)
				A(rv,cv) = Z
				rhs(rv) = vZ
				for(v2 <- vertices(v)) {
					val rv2 = MyLS.row(v2)
					val cv2 = MyLS.col(v2)
					A(rv,cv2) = Z
					A(rv2,cv) = Z
				}
			}
			
			MyLS.solve(A,delta_v,rhs)
			
			for(v <- interior) {
				val c = MyLS.col(v)
				Velocity(v) += delta_v(c)
			}
			for(v <- interior) {
				Position(v) += h*Velocity(v)
			}
			
			t += h
			
			print()
			
		}
		
		/*
		
		
		
		
		for(v <- vertices(mesh)) {
			Print(position(v))
			val row = MyLS.row(v)
			val col = MyLS.col(v)
			val sz = size(vertices(v))
			val I = Mat( Vec(1.0,0.0,0.0),
				         Vec(0.0,1.0,0.0),
				         Vec(0.0,0.0,1.0))
			if(sz < 6) {
				A(row,col) = I
				b(row) = position(v)
			} else {
				A(row,col) = -I
				for(v2 <- vertices(v)) {
					A(row,MyLS.col(v2)) = I / sz
				}
				b(row) = Vec(0.0,0.0,0.0)
			}
		}
		
		MyLS.solve(A,x,b)
		for(v <- vertices(mesh)) {
			val col = MyLS.col(v)
			if(size(vertices(v)) == 6) {
				Print("ID ", ID(v), " value ", x(col), position(v))
				vdb_point(position(v).x,position(v).y,position(v).z)
			}
		}*/
	}
}