import "ebb"
local L = require "ebblib" -- Every Ebb File should start with this command

error("OUT OF DATE EXAMPLE")

local Grid  = require 'ebb.domains.grid'
local PN    = require 'ebb.lib.pathname'
local cmath = terralib.includecstring '#include <math.h>'
local vdb   = require 'ebb.lib.vdb'


local N = 40
local init_height = 3.5
local width = 10.0

local sphere_radius = 3
local sphere_center = L.Global(L.vec3d, {0,0,0})
local collision_spring_k = 100

------------------------------------------------------------------------------

-- the grid's own coordinate system doesn't really matter
-- We just make sure it's dimensioned sensibly
local grid = Grid.NewGrid2d{
    size   = {N,N},
    origin = {0,0},
    width  = {width,width},
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
--grid.edges:NewField('J_x', L.vec3d):Load({0,0,0})
--grid.edges:NewField('J_y', L.vec3d):Load({0,0,0})
--grid.edges:NewField('J_z', L.vec3d):Load({0,0,0})
grid.vertices:NewField('J_diag', L.vec3d):Load({0,0,0})

local init_fields = ebb (v : grid.vertices)
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

local dt = L.Global(L.double, 0.00005)

local ebb spring_dvel(dir)
  var dir_len = L.length(dir)
  var stretch = (IDEAL_LEN - dir_len) / dir_len
  return SPRING_K * stretch * dir
end

local accel_interior = ebb (v : grid.vertices)
  v.dvel = { 0, 0, 0 }
  v.dpos = v.vel

  -- gravity
  v.dvel += { 0, 0, -GRAVITY }

  -- spring force to neighbors
  v.dvel +=
    spring_dvel(v.pos - v(-1,0).pos) +
    spring_dvel(v.pos - v( 1,0).pos) +
    spring_dvel(v.pos - v(0,-1).pos) +
    spring_dvel(v.pos - v(0, 1).pos)
end

local accel_boundary = ebb (v : grid.vertices)
  v.dvel = { 0, 0, 0 }
  v.dpos = v.vel

  -- gravity
  v.dvel += { 0, 0, -GRAVITY }

  -- spring force to neighbors
  if v.xneg_depth == 0 then
    v.dvel += spring_dvel(v.pos - v(-1,0).pos)
  end
  if v.xpos_depth == 0 then
    v.dvel += spring_dvel(v.pos - v(1,0).pos)
  end
  if v.yneg_depth == 0 then
    v.dvel += spring_dvel(v.pos - v(0,-1).pos)
  end
  if v.ypos_depth == 0 then
    v.dvel += spring_dvel(v.pos - v(0,1).pos)
  end
end

local accel_collisions = ebb (v : grid.vertices)
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
    if L.dot(v.dvel,sphere_normal) < 0 then
      v.dvel -= L.dot(v.dvel,sphere_normal)*sphere_normal
    end
    if L.dot(v.dpos,sphere_normal) < 0 then
      v.dpos -= L.dot(v.dpos,sphere_normal)*sphere_normal
    end
    -- apply some degree of friction?
    v.dvel += -FRICTION * v.dpos

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

  -- timescale & update
  v.pos += dt * v.dpos
  v.vel += dt * v.dvel
end

local compute_acceleration = ebb (v : grid.vertices)
  var dvel : L.vec3d = { 0, 0, 0 }
  var dpos : L.vec3d = v.vel

  -- gravity
  dvel += { 0, 0, -GRAVITY }

  -- spring force to neighbors
  if v.xneg_depth == 0 then
    var ldir = v.pos - v(-1,0).pos
    dvel += spring_dvel(ldir)
  end
  if v.xpos_depth == 0 then
    var rdir = v.pos - v(1,0).pos
    dvel += spring_dvel(rdir)
  end
  if v.yneg_depth == 0 then
    var ddir = v.pos - v(0,-1).pos
    dvel += spring_dvel(ddir)
  end
  if v.ypos_depth == 0 then
    var udir = v.pos - v(0,1).pos
    dvel += spring_dvel(udir)
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


local apply_update = ebb (v : grid.vertices)
  v.pos += v.dpos
  v.vel += v.dvel
end

------------------------------------------------------------------------------


local sqrt3 = math.sqrt(3)
local ebb draw_cloth(v : grid.vertices)
  
  if v.xpos_depth == 0 and v.ypos_depth == 0 then
    var p00 = v(0,0).pos
    var p01 = v(1,0).pos
    var p10 = v(0,1).pos
    var p11 = v(1,1).pos

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
local steps_per_frame   = math.floor((1/fps)/dt:get())
local total_steps       = steps_per_frame * fps * total_sec

for i=1,(total_steps+2) do
  if i % steps_per_frame == 1 then
    vdb.vbegin()
      vdb.frame()
      draw_cloth(grid.vertices)
    vdb.vend()
    --io.read()
  end

  accel_boundary(grid.vertices.boundary)
  accel_interior(grid.vertices.interior)
  accel_collisions(grid.vertices)
  --compute_acceleration(grid.vertices)
  --apply_update(grid.vertices)

end






