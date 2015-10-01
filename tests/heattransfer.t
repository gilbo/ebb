import 'ebb.liszt'

local M     = {
  vertices  = L.NewRelation { name="vertices",  size=8  },
  edges     = L.NewRelation { name="edges",     size=12 },
}
M.edges:NewField('head', M.vertices):Load({
  1, 2, 3, 0,   5, 6, 7, 4,   5, 0, 6, 7
})
M.edges:NewField('tail', M.vertices):Load({
  0, 1, 2, 3,   4, 5, 6, 7,   1, 4, 2, 3
})
M.vertices:NewField('position', L.vec3d):Load({
  {0, 0, 0},
  {1, 0, 0},
  {1, 1, 0},
  {0, 1, 0},
  {0, 0, 1},
  {1, 0, 1},
  {1, 1, 1},
  {0, 1, 1},
})

local init_to_zero = terra (mem : &float, i : int) mem[0] = 0 end
local function init_temp (i)
  if i == 0 then
    return 1000
  else
    return 0
  end
end

M.vertices:NewField('flux',        L.float):Load(0)
M.vertices:NewField('jacobistep',  L.float):Load(0)
M.vertices:NewField('temperature', L.float):Load(init_temp)

local liszt compute_step (e : M.edges)
  var v1   = e.head
  var v2   = e.tail
  var dp   = L.vec3f(v1.position - v2.position)
  var dt   = v1.temperature - v2.temperature
  var step = 1.0f / L.length(dp)

  v1.flux += -dt * step
  v2.flux +=  dt * step

  v1.jacobistep += step
  v2.jacobistep += step
end

local liszt propagate_temp (p : M.vertices)
  p.temperature += L.float(.01) * p.flux / p.jacobistep
end

local liszt clear (p : M.vertices)
  p.flux       = 0
  p.jacobistep = 0
end

for i = 1, 1000 do
  M.edges:foreach(compute_step)
  M.vertices:foreach(propagate_temp)
  M.vertices:foreach(clear)
end

M.vertices.temperature:print()
