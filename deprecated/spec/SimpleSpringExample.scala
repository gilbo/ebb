import Liszt.Language._
import Liszt.MetaInteger._

@lisztcode
object SSS {
// 
// var deltat = 0.15f
// var maxforce = 0.0f
// def dampedSpringForce(L : Float3, W : Float3) : Float3 = {
// 	val l = length(L)
// 	return -(L/l) * (Ks*(l-rl) + Kd*(dot(L,W)/l))
// }
// 
// //================================================================================
// 
// val rl = 1; val Ks = 20.0f; val Kd = 50.0f
// val Position = FieldWithLabel[Vertex,Float3]("position")
// val Velocity = FieldWithConst[Vertex,Float3](Vec(0.f,0.f,0.f))
// val Force = FieldWithConst[Vertex,Float3](Vec(0.f,0.f,0.f))
// 
// //================================================================================
// 
// def main() {
// 
// var t = 0.f;
// while (t < 2.0) {
// for (spring <- edges(mesh)) {
// 	val v1 = head(spring)
// 	val v2 = tail(spring)
// 	val L = Position(v1) - Position(v2)
// 	val W = Velocity(v1) - Velocity(v2)
// 	val springForce = dampedSpringForce(L,W)
// 	Force(v1) += springForce
// 	Force(v2) -= springForce
// 	//maxforce = max(maxforce, springForce)
// }
// deltat = 1 / maxforce*0.5f 
// for (ptcl <- vertices(mesh)) {
// 	Velocity(ptcl) += deltat * Force(ptcl)
// }
// t += deltat
// for (ptcl <- vertices(mesh)) { Force(ptcl) = Vec(0.f,0.f,0.f) }
// }
// 	
// }
//================================================================================
}