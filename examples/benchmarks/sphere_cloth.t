import "compiler.liszt" -- Every Liszt File should start with this command

local Grid  = L.require 'domains.grid'
local PN    = L.require 'lib.pathname'
local cmath = terralib.includecstring '#include <math.h>'
local vdb   = L.require 'lib.vdb'


local N = 40
local init_height = 3.5
local width = 10.0

local sphere_radius = 3
local sphere_center = L.NewGlobal(L.vec3d, {0,0,0})
local collision_spring_k = 100

------------------------------------------------------------------------------

-- the grid's own coordinate system doesn't really matter
-- We just make sure it's dimensioned sensibly
local grid = Grid.New2dUniformGrid{
    size   = {N,N},
    origin = {0,0},
    width  = width,
    height = width,
}

------------------------------------------------------------------------------

grid.vertices:NewField('pos', L.vec3d):Load({0,0,0})
grid.vertices:NewField('vel', L.vec3d):Load({0,0,0})
grid.vertices:NewField('dvel', L.vec3d):Load({0,0,0})
grid.vertices:NewField('dpos', L.vec3d):Load({0,0,0})

grid.vertices:NewField('b_vel', L.vec3d):Load({0,0,0})
grid.vertices:NewField('b_pos', L.vec3d):Load({0,0,0})
grid.vertices:NewField('dpos_temp', L.vec3d):Load({0,0,0})

-- Jacobian matrix...
grid.edges:NewField('J_x', L.vec3d):Load({0,0,0})
grid.edges:NewField('J_y', L.vec3d):Load({0,0,0})
grid.edges:NewField('J_z', L.vec3d):Load({0,0,0})
grid.vertices:NewField('J_diag', L.vec3d):Load({0,0,0})

local init_fields = liszt kernel(v : grid.vertices)
  var i = L.int(v.xid)
  var j = L.int(v.yid)

  var x = width * ((i/L.double(N)) - 0.5)
  var y = width * ((j/L.double(N)) - 0.5)

  v.pos = { x, y, init_height }
  --v.vel = { 0, 0, 0 }
end

init_fields(grid.vertices)



------------------------------------------------------------------------------

local GRAVITY = 0.98 -- look up meaning...
local IDEAL_LEN = width / N
local SPRING_K = 1000.0
local FRICTION = 2.0

local dt = L.NewGlobal(L.double, 0.00005)

local spring_dvel = L.NewMacro(function(dir)
  return liszt quote
    var dir_len = L.length(dir)
    var stretch = (IDEAL_LEN - dir_len) / dir_len
  in
    SPRING_K * stretch * dir
  end
end)

local compute_acceleration = liszt kernel(v : grid.vertices)
  var dvel : L.vec3d = { 0, 0, 0 }
  var dpos : L.vec3d = v.vel

  -- gravity
  dvel += { 0, 0, -GRAVITY }

  -- spring force to neighbors
  if v.has_left then
    var ldir = v.pos - v.left.pos
    dvel += spring_dvel(ldir)
  end
  if v.has_right then
    var rdir = v.pos - v.right.pos
    dvel += spring_dvel(rdir)
  end
  if v.has_up then
    var udir = v.pos - v.up.pos
    dvel += spring_dvel(udir)
  end
  if v.has_down then
    var ddir = v.pos - v.down.pos
    dvel += spring_dvel(ddir)
  end

  -- collision penalty
  --var dir = v.pos - sphere_center
  --var lendir = L.length(dir)
  --if lendir < sphere_radius then
  --  var stretch = (sphere_radius - lendir) / lendir
  --  dvel += collision_spring_k * stretch * dir
  --end

  -- collision fix
  var sphere_normal = v.pos-sphere_center
  var sphere_dist = L.length(sphere_normal)
  if sphere_dist < sphere_radius then
    sphere_normal = sphere_normal/sphere_dist
    var offset = (sphere_radius - sphere_dist) * sphere_normal
    -- pull out the normal component
    if L.dot(dvel,sphere_normal) < 0 then
      dvel -= L.dot(dvel,sphere_normal)*sphere_normal
    end
    if L.dot(dpos,sphere_normal) < 0 then
      dpos -= L.dot(dpos,sphere_normal)*sphere_normal
    end
    -- apply some degree of friction?
    dvel += -FRICTION * dpos

    -- offset the vertex to sit at the sphere surface
    --dpos +=  + (offset / dt)

    --if v.pos[2] < 0 then
    --  dpos[2] = 0
    --  dvel[2] = 0
    --else
    --  dpos = {0,0,0}
    --  dvel = {0,0,0}
    --end
  end

  v.dvel = dt * dvel
  v.dpos = dt * dpos
end


local apply_update = liszt kernel(v : grid.vertices)
  v.pos += v.dpos
  v.vel += v.dvel
end

------------------------------------------------------------------------------


local sqrt3 = math.sqrt(3)
local draw_cloth = liszt kernel (v : grid.vertices)
  
  if v.has_right and v.has_up then
    var p00 = v.pos
    var p01 = v.right.pos
    var p10 = v.up.pos
    var p11 = v.right.up.pos

    var norm = L.cross(p01-p00, p11-p01) +
               L.cross(p11-p01, p10-p11) +
               L.cross(p10-p11, p00-p10) +
               L.cross(p00-p10, p01-p00)
    var d = L.dot({1/sqrt3, 1/sqrt3, 1/sqrt3}, norm / L.length(norm))
    if d > 1.0 then  d = 1.0  end
    if d < -1.0 then d = -1.0 end
    var val = d * 0.5 + 0.5
    var col = {val,val,val}
    vdb.color(col)

    vdb.triangle(p00,p01,p11)
    vdb.triangle(p00,p11,p10)
  end
--  var color = {1,1,1}
--  vdb.color(color)
--
--  var pos = v.pos
--
--  --if (L.id(v) < 5) then
--  --  L.print(pos)
--  --end
--
--  vdb.point(pos)
end

local total_sec         = 15
local fps               = 30
local steps_per_frame   = math.floor((1/fps)/dt:value())
local total_steps       = steps_per_frame * fps * total_sec

for i=1,(total_steps+2) do
  if i % steps_per_frame == 1 then
    vdb.vbegin()
      vdb.frame()
      draw_cloth(grid.vertices)
    vdb.vend()
    --io.read()
  end

  compute_acceleration(grid.vertices)
  apply_update(grid.vertices)

end






