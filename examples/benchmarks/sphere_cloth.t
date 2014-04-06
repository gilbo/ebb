import "compiler.liszt" -- Every Liszt File should start with this command

local Grid = terralib.require 'compiler.grid'
local PN = terralib.require 'compiler.pathname'
local cmath = terralib.includecstring '#include <math.h>'
local vdb = terralib.require 'compiler.vdb'


local N = 40
local init_height = 3.5
local width = 10.0

local sphere_radius = 3
local sphere_center = L.NewGlobal(L.vec3d, {0,0,0})
local collision_spring_k = 100

------------------------------------------------------------------------------

-- the grid's own coordinate system doesn't really matter
-- We just make sure it's dimensioned sensibly
local grid = Grid.New2dUniformGrid(N,N, {0,0}, width, width)

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
  var i = L.int(v.xy_ids[0])
  var j = L.int(v.xy_ids[1])

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


-- implicit cloth

-- We have a function F(x,v)
-- Then, we want to integrate
-- d(x,v)/dt = F(x,v)
-- (x1,v1)-(x0,v0) = dt*F(x1,v1)
-- [I - dt*F](x1,v1) = (x0,v0)

-- What is F?  Doing a bunch of math elsewhere, I figured out that...
-- J[F_i](y) = SUM_j [(L/E^3)*OUTER(e_ij,e_ij) + (L-E)/E * 1] * (y_i-y_j)
-- where J[F_i] is the Jacobian for output vertex i's velocity
--
-- [I - dt*F](x0+dx, v0+dv) = (x0,v0)
-- [I - dt*J](dx,dv) = (x0,v0) - [I-dt*F](x0,v0)
--                   = -dt*F(x0,v0)
-- 

local edge_stretch = L.NewMacro(function(dir)
  return liszt quote
    var dir_len = L.length(dir)
    var stretch : L.double = (IDEAL_LEN - dir_len) / dir_len
  in
    SPRING_K * stretch
  end
end)
local edge_dstretch = L.NewMacro(function(dir)
  return liszt quote
    var dir_len = L.length(dir)
    var dstretch : L.double = IDEAL_LEN / dir_len
  in
    SPRING_K * dstretch
  end
end)

local jacobi_write = L.NewMacro(function(edge, dir)
  return liszt quote
    var stretch  = edge_stretch(dir)
    var dstretch = edge_dstretch(dir)
    var jx = dt * (dstretch * dir[0]*dir + {stretch,0,0})
    var jy = dt * (dstretch * dir[1]*dir + {0,stretch,0})
    var jz = dt * (dstretch * dir[2]*dir + {0,0,stretch})
    edge.J_x = jx
    edge.J_y = jy
    edge.J_z = jz
  in
    { jx[0], jy[1], jz[2] }
  end
end)

local jacobi_diag = L.NewMacro(function(edge)
  return liszt ` { edge.J_x[0], edge.J_y[1], edge.J_z[2] }
end)

-- We can do this wrong, but maybe it's good enough?
local implicit_setup = liszt kernel(v : grid.vertices)
  v.b_vel = -v.dvel
  v.b_pos = -v.dpos
  var diag = L.vec3d({0,0,0})

  -- compute the Jacobian and store it on the edges
  if v.has_left then
    var ldir = v.pos - v.left.pos
    diag += jacobi_write(v.left_edge, ldir)
  end
  if v.has_right then
    var rdir = v.pos - v.right.pos
    diag += jacobi_write(v.right_edge, rdir)
  end
  if v.has_up then
    var udir = v.pos - v.up.pos
    diag += jacobi_write(v.up_edge, udir)
  end
  if v.has_down then
    var ddir = v.pos - v.down.pos
    diag += jacobi_write(v.down_edge, ddir)
  end

  v.J_diag = {1,1,1} - diag
end

local jacobi_edge_diag = L.NewMacro(function(v, dir)
  return liszt `{ v.J_diag[0] * dir[0],
                  v.J_diag[1] * dir[1],
                  v.J_diag[2] * dir[2] }
end)

local jacobi_edge = L.NewMacro(function(edge, dir)
  return liszt quote
    var x = L.dot(edge.J_x, dir)
    var y = L.dot(edge.J_y, dir)
    var z = L.dot(edge.J_z, dir)
  in
    { x, y, z }
  end
end)

-- We apply the following Jacobi iteration here:
-- A = D+R, so R = A-D
-- x_new = D^{-1} * (b - R * x_old) = D^{-1} * (b - A * x_old + D * x_old)
-- solve for new dpos given b_vel
local jacobi_step = liszt kernel(v : grid.vertices)
  -- b
  var sum = v.b_vel

  -- b - A * x_old
  if v.has_left then
    var ldir = v.dpos - v.left.dpos
    sum -= jacobi_edge(v.left_edge, ldir)
  end
  if v.has_right then
    var rdir = v.dpos - v.right.dpos
    sum -= jacobi_edge(v.right_edge, rdir)
  end
  if v.has_up then
    var udir = v.dpos - v.up.dpos
    sum -= jacobi_edge(v.up_edge, udir)
  end
  if v.has_down then
    var ddir = v.dpos - v.down.dpos
    sum -= jacobi_edge(v.down_edge, ddir)
  end

  -- b - A * x_old + D * x_old
  sum += { v.J_diag[0] * v.dpos[0],
           v.J_diag[1] * v.dpos[1],
           v.J_diag[2] * v.dpos[2] }

  -- times D^{-1}
  sum[0] = sum[0] / v.J_diag[0]
  sum[1] = sum[1] / v.J_diag[1]
  sum[2] = sum[2] / v.J_diag[2]

  v.dpos_temp = sum
end

local commit_jacobi = liszt kernel(v : grid.vertices)
  v.dpos = v.dpos_temp
end

-- want to set v1 = v0 + dvel = dpos
-- So, we get dvel = dpos - v0
local set_dvel = liszt kernel(v : grid.vertices)
  v.dvel = v.dpos - v.vel
end

local function implicit_acceleration(vertices)
  compute_acceleration(vertices)
  implicit_setup(vertices)

  for i=1,20 do
    jacobi_step(vertices)
    commit_jacobi(vertices)
  end

  set_dvel(vertices)
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
  --implicit_acceleration(grid.vertices)
  apply_update(grid.vertices)

end






