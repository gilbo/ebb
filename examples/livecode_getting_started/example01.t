import "compiler.liszt"

local ioOff = L.require 'domains.ioOff'
local mesh  = ioOff.LoadTrimesh(
  'examples/livecode_getting_started/octa.off')

print(mesh.vertices:Size())
print(mesh.triangles:Size())


----------------------------

mesh.vertices.pos:print()
mesh.triangles.v:print()


----------------------------

-- At this point open up the OFF file to show them TADA.

----------------------------

-- Then, let's try translating the mesh vertices

local liszt translate ( v : mesh.vertices )
  v.pos += {1,0,0}
end

mesh.vertices:map(translate)

mesh.vertices.pos:print()