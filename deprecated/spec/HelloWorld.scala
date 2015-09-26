import Liszt.Language._

@lisztcode
object Main {
  val position = FieldWithLabel[Vertex,Vec[_3,Float]]("position")
  val field = FieldWithConst[Cell,Vec[_2,Int]](Vec(0,0))
  val field2 = FieldWithConst[Cell,Int](0)
  val myid = FieldWithConst[Vertex,Int](0)//,Vec[_2,Int]](Vec(0,0))
  def main() {
  	Print("hello")
  	
  	var a = Vec(1,2)
  	for(c <- cells(mesh)) {
  		Print(c)
  		field(c) = Vec(ID(c),ID(c)+1)
  		for(v <- vertices(c))
  			Print(position(v))
  	}
  	for(c <- cells(mesh)) {
  		Print(c,field(c))
  		Print(a)
  	}
  	Print(a)
  	for(c <- cells(mesh)) {
  		a += Vec(1,2)
  	}
  	Print(a)
  	for(v <- vertices(mesh)) {
  		myid(v) += ID(v) //Vec(ID(v),ID(v)+1)
  	}
  	for(c <- cells(mesh)) {
  		for(v <- vertices(c)) {
  			Print(v," ",myid(v))
  		}
  	}
  	
  	for(v <- vertices(mesh)) {
  		for(c <- cells(v)) {
  			field2(c) += 1
  		}
  	}
  	for(c <- cells(mesh)) {
  		Print(c,field2(c))
  	}
  }
}