import "compiler.liszt"
--L.default_processor = L.GPU

local Grid  = L.require 'domains.grid'
local cmath = terralib.includecstring [[
#include <math.h>
#include <stdlib.h>
#include <time.h>

int nancheck(float val) {
    return isnan(val);
}

float rand_float()
{
      float r = (float)rand() / (float)RAND_MAX;
      return r;
}
]]
cmath.srand(cmath.time(nil));
local vdb   = L.require 'lib.vdb'

local N = 32
local width = N
local origin = { -width/2, -width/2, -width/2 }
local grid = Grid.NewGrid3d {
  size    = {N, N, N},
  origin  = origin,
  width   = {width, width, width}
}

local xcw, ycw, zcw = grid:xCellWidth(), grid:yCellWidth(), grid:zCellWidth()


-- quads

local quads = L.NewRelation {
    mode = 'ELASTIC',
    size = 0,
    name = 'quads',
}
-- just use the grid's cells as surrogates for vertices
quads:NewField('v0', grid.cells):Load(0)
quads:NewField('v1', grid.cells):Load(0)
quads:NewField('v2', grid.cells):Load(0)
quads:NewField('v3', grid.cells):Load(0)


local clamp = liszt function(val,minval,maxval)
  if val > maxval then val = maxval end
  if val < minval then val = minval end
  return val
end


-------------------------------
-- Field Declarations
-------------------------------

grid.vertices:NewField('pos', L.vec3f):Load(function(i)
  local xi = i%(N+1)
  local xr = (i-xi)/(N+1)
  local yi = xr%(N+1)
  local yr = (xr-yi)/(N+1)
  local zi = yr

  return {
    origin[1] + xi * xcw,
    origin[2] + yi * ycw,
    origin[3] + zi * zcw,
  }
end)
grid.vertices:NewField('scalar', L.float):Load(1)
grid.vertices:NewField('gradient', L.vec3f):Load({0,0,0})

-- where between the tail and head the cut happens,
-- value outside [0,1] means no cut
grid.edges:NewField('tval', L.float):Load(-1)
grid.edges:NewField('is_cut', L.bool):Load(false)
grid.edges:NewField('cut_pos', L.vec3f):Load({0,0,0})
grid.edges:NewField('surf_normal', L.vec3f):Load({0,0,0})
grid.cells:NewField('dual_pos', L.vec3f):Load({0,0,0})


-------------------------------
-- Scalar Field Options
-------------------------------
local set_sphere_field = liszt kernel( v : grid.vertices )
  var origin = L.vec3f({0,0,0})
  var r = v.pos - origin
  var radius = L.float(width * 0.75)
  var radius2 = radius*radius

  v.scalar = L.dot(r,r) - radius2
  v.gradient = 2 * r -- gradient
end

local set_hyperboloid = liszt kernel( v : grid.vertices )
  var origin = L.vec3f({0,0,0})
  var p = v.pos - origin

  -- f(x,y,z) = x^2 - y^2 + z^2
  -- grad(f)(x,y,z) = (2x, -2y, 2z)
  v.scalar = p[0]*p[0] - p[1]*p[1] + p[2]*p[2] + 10
  v.gradient = 2 * p
  v.gradient[1] = -v.gradient[1]
end

local set_sins = liszt kernel( v : grid.vertices)
  var origin = L.vec3f({0,0,0})
  var p = v.pos - origin

  -- f(x,y,z) = 3*sin(0.5*sqrt(x^2 + y^2 + 0.1))
  var r2 = p[0]*p[0] + p[1]*p[1] + 0.1
  var dr2_dx = 2*p[0]
  var dr2_dy = 2*p[1]
  var sqrt = cmath.sqrt(r2)
  var dsqrt_dx = 0.5/sqrt * dr2_dx
  var dsqrt_dy = 0.5/sqrt * dr2_dy
  var fscalar = L.float(3*cmath.sin(0.5*sqrt))
  var dsin_dx = cmath.cos(0.5*sqrt) * 0.5*dsqrt_dx
  var dsin_dy = cmath.cos(0.5*sqrt) * 0.5*dsqrt_dy
  var fgrad   = { L.float(3*dsin_dx), L.float(3*dsin_dy), 0 }

  -- g(x,y,z) = 2.5*sin(0.3*(x+y+z))
  var t = 0.3*(p[0]+p[1]+p[2])
  var gscalar = L.float(2.5*cmath.sin(t))
  var dsin = L.float( 2.5*cmath.cos(t) * 0.3 )
  var ggrad   = { dsin, dsin, dsin }

  -- h(x,y,z) = -2 * cos(0.2 * z)
  var hscalar = L.float(-2 * cmath.cos(0.2*p[2]))
  var hgrad = {0,0, L.float(2 * cmath.sin(0.2*p[2]) * 0.2) }

  v.scalar = fscalar + gscalar + hscalar
  v.gradient = fgrad + ggrad + hgrad
end



-------------------------------
-- Compute the contouring
-------------------------------
local cut_edges = liszt kernel( e : grid.edges )
  -- guard against boundary cases?

  var head = e.head
  var tail = e.tail

  -- v_0 is value at the tail, v_1 is value at the head
  -- v(t) = v_0 + t * (v1-v0)
  -- for v(t) = 0, t = -v0 / (v1-v0)
  var t = - tail.scalar / (head.scalar - tail.scalar)
  e.tval = t
  var is_cut = t >= 0 and t <= 1
  if t == 0 then
    is_cut = head.scalar > 0
  elseif t == 1 then
    is_cut = tail.scalar > 0
  end
  e.is_cut = is_cut



  -- normal
  if is_cut then
    e.cut_pos = tail.pos + t * (head.pos - tail.pos)
    e.surf_normal = tail.gradient + t * (head.gradient - tail.gradient)
    e.surf_normal = e.surf_normal / L.length(e.surf_normal)
  end

  -- emit quads if appropriate
  if is_cut then
    var a0 : L.addr
    var a1 : L.addr
    var a2 : L.addr
    var a3 : L.addr

    var on_xbd = tail.xneg_depth > 0 or tail.xpos_depth > 0
    var on_ybd = tail.yneg_depth > 0 or tail.ypos_depth > 0
    var on_zbd = tail.zneg_depth > 0 or tail.zpos_depth > 0

    var base = tail.cell
    var do_emit = false

    if e.is_x then
      if not on_ybd and not on_zbd then
        do_emit = true
        a0 = L.id(base( 0, 0, 0))
        a1 = L.id(base( 0,-1, 0))
        a2 = L.id(base( 0,-1,-1))
        a3 = L.id(base( 0, 0,-1))
      end
    elseif e.is_y then
      if not on_xbd and not on_zbd then
        do_emit = true
        a0 = L.id(base( 0, 0, 0))
        a1 = L.id(base( 0, 0,-1))
        a2 = L.id(base(-1, 0,-1))
        a3 = L.id(base(-1, 0, 0))
      end
    else -- z aligned
      if not on_xbd and not on_ybd then
        do_emit = true
        a0 = L.id(base( 0, 0, 0))
        a1 = L.id(base(-1, 0, 0))
        a2 = L.id(base(-1,-1, 0))
        a3 = L.id(base( 0,-1, 0))
      end
    end

    if do_emit then
      if tail.scalar > 0 then
        var temp = a1
        a1 = a3
        a3 = temp
      end
      var c0 = L.UNSAFE_ROW(a0, grid.cells)
      var c1 = L.UNSAFE_ROW(a1, grid.cells)
      var c2 = L.UNSAFE_ROW(a2, grid.cells)
      var c3 = L.UNSAFE_ROW(a3, grid.cells)

      insert { v0=c0, v1=c1, v2=c2, v3=c3 } into quads
    end
  end
end


-- I'm concerned about the stability of this routine...
local cholesky3x3 = liszt function(diag, offdiag, rhs)
  -- compute LDL decomposition with 1s along diagonal of L
  var D0  = diag[0]
  var L10 = (1/D0) * offdiag[0]
  var D1  = diag[1] - L10*L10*D0
  var L20 = (1/D0) * offdiag[1]
  var L21 = (1/D1) * (offdiag[2] - L20*L10*D0)
  var D2  = diag[2] - L20*L20*D0 - L21*L21*D1

  -- now we need to solve  LDLX = RHS in pieces
  var DLX0 = rhs[0]
  var DLX1 = rhs[1] - L10*DLX0
  var DLX2 = rhs[2] - L20*DLX0 - L21*DLX1
  var LX0 = DLX0 / D0
  var LX1 = DLX1 / D1
  var LX2 = DLX2 / D2
  var X2 = LX2
  var X1 = LX1 - L21*X2
  var X0 = LX0 - L20*X2 - L10*X1

  return L.vec3f({ X0, X1, X2 })
end

local householder3x3 = liszt function(diag, offdiag, rhs)
  -- write out A explicitly (columns here)
  var A0 = {diag[0], offdiag[0], offdiag[1]}
  var A1 = {offdiag[0], diag[1], offdiag[2]}
  var A2 = {offdiag[1], offdiag[2], diag[2]}

  -- start with X = A0
  var alpha0 = L.float(L.length(A0))
  if A0[0] < 0 then alpha0 = -alpha0 end
  var U0 = A0
      U0[0] -= alpha0
  var V0 = U0 / L.length(U0)
  -- Q0 = I - 2*outer(V0,V0)
  var R00 = alpha0
  A1 -= V0*(2*L.dot(V0,A1)) -- = Q0*A1
  A2 -= V0*(2*L.dot(V0,A2)) -- = Q0*A2
  var R01 = A1[0]
  var R02 = A2[0]

  -- also go ahead and apply Q0 to RHS immediately note Q0 is symmetric
  var Q0RHS = rhs - V0*(2*L.dot(V0,rhs))

  -- restrict down to 2x2 minor
  var A1m = {A1[1], A1[2]}
  var A2m = {A2[1], A2[2]}

  -- now, we have X = A1, but can ignore the first coordinate
  var alpha1 = L.float(L.length(A1m))
  if A1m[0] < 0 then alpha1 = -alpha1 end
  var U1 = A1m
      U1[0] -= alpha1
  var V1 = U1 / L.length(U1)
  -- Q1 = I - 2*outer(V1,V1)
  var R11 = alpha1
  A1m -= V1*(2*L.dot(V1,A1m)) -- = Q1*A1m
  A2m -= V1*(2*L.dot(V1,A2m)) -- = Q1*A2m
  var R12 = A2m[0]

  -- go ahead and apply Q1 to RHS immediately
  var q0rhsProj = {Q0RHS[1], Q0RHS[2]}
  q0rhsProj -= V1*(2*L.dot(V1,q0rhsProj))
  -- here's the full application of Q to the right hand side
  var Qrhs = {Q0RHS[0], q0rhsProj[0], q0rhsProj[1]}

  var R22 = A2m[1]

  -- now we just need to back substitute
  var X2 = Qrhs[2] / R22
  var X1 = (Qrhs[1] - X2*R12) / R11
  var X0 = (Qrhs[0] - X1*R01 - X2*R02) / R00

  return {X0, X1, X2}
end

local cubeclamp = liszt function(vec,minvec,maxvec)
  for i=0,3 do
    vec[i] = clamp(vec[i], minvec[i], maxvec[i])
  end
  return vec
end

-- we want to place the dual vertex so that it minimizes a quadratic error:
-- SUM dot( surf_normal_i,  X - cut_pos_i )^2
-- = SUM ( dot(n_i,X) - dot(n_i,cut_i) )^2
-- = SUM dot(n_i,X)^2 - 2 * dot(n_i,X) * dot(n_i,cut_i) + dot(n_i,cut_i)^2
-- which is minimized when
-- 0 = SUM n_i * dot(n_i,X)  -  n_i * dot(n_i,cut_i)
-- SUM (n_i * dot(n_i,cut_i)) = SUM n_i * dot(n_i,X)
-- which is a symmetric SPD linear system.  To ensure non-singularity,
-- we can add a small regularization term with an epsilon scale factor

local gather_dual_pos = liszt kernel( c : grid.cells )
  var v = c.vertex

  var n_cut = 0
  var rhs  = L.vec3f({0,0,0})
  var diag = L.vec3f({0,0,0})
  var offdiag = L.vec3f({0,0,0}) -- pattern 0,1; 0,2; 1,2

  var centroid = L.vec3f({0,0,0})

  -- xedges
  for i=0,2 do for j=0,2 do
    var e = v(0,i,j).xedge

    if e.is_cut then
      n_cut += 1
      var n = e.surf_normal
      var p = e.cut_pos - c.center -- recenter

      centroid  += p
      rhs       += n * L.dot(n, p)
      diag      += { n[0]*n[0], n[1]*n[1], n[2]*n[2] }
      offdiag   += { n[0]*n[1], n[0]*n[2], n[1]*n[2] }
    end
  end end

  -- yedges
  for i=0,2 do for j=0,2 do
    var e = v(i,0,j).yedge
    
    if e.is_cut then
      n_cut += 1
      var n = e.surf_normal
      var p = e.cut_pos - c.center -- recenter

      centroid  += p
      rhs       += n * L.dot(n, p)
      diag      += { n[0]*n[0], n[1]*n[1], n[2]*n[2] }
      offdiag   += { n[0]*n[1], n[0]*n[2], n[1]*n[2] }
    end
  end end

  -- zedges
  for i=0,2 do for j=0,2 do
    var e = v(i,j,0).zedge
    
    if e.is_cut then
      n_cut += 1
      var n = e.surf_normal
      var p = e.cut_pos - c.center -- recenter

      centroid  += p
      rhs       += n * L.dot(n, p)
      diag      += { n[0]*n[0], n[1]*n[1], n[2]*n[2] }
      offdiag   += { n[0]*n[1], n[0]*n[2], n[1]*n[2] }
    end
  end end

  if n_cut > 0 then
    centroid = centroid / n_cut

    -- regularization for safety
    var epsilon = L.float(0.01)
    diag += {epsilon, epsilon, epsilon}
    rhs  += epsilon * centroid

    -- Do the 3x3 solve
    c.dual_pos = cholesky3x3(diag, offdiag, rhs)
    --c.dual_pos = householder3x3(diag, offdiag, rhs)

    -- Note that the Lossazo paper suggests a different solution
    -- method that they claim is more stable.  This seems ok for demo

    -- clamp for safety
    var halfwidth = { L.float(xcw), L.float(ycw), L.float(zcw) } / 2
    c.dual_pos = cubeclamp(c.dual_pos, -halfwidth, halfwidth)

    -- readjust back into global coordinates
    c.dual_pos += c.center
  end
end





-------------------------------
-- visualize
-------------------------------
local sqrt3 = math.sqrt(3)
local lightdir = L.NewVector(L.float, {1/sqrt3, 1/sqrt3, 1/sqrt3})
local trinorm = liszt function(p0,p1,p2)
  var n = L.vec3f(L.cross(p1-p0, p2-p0))
  var len = L.length(n)
  --if n < 0.00000001 then  n = {0,0,0}
  --else                    n = n / len
  --end
  return n / len
end

local debug_draw_quads = liszt kernel( q : quads )
  var p0 = q.v0.dual_pos
  var p1 = q.v1.dual_pos
  var p2 = q.v2.dual_pos
  var p3 = q.v3.dual_pos

  var n1 = trinorm(p0,p1,p2)
  var n2 = trinorm(p0,p2,p3)

  var d1 = clamp(L.dot(lightdir, n1), -1, 1) * 0.5 + 0.5
  var d2 = clamp(L.dot(lightdir, n2), -1, 1) * 0.5 + 0.5
  var c1 = {d1, d1, d1}
  var c2 = {d2, d2, d2}

  var guard = p0[0] > 0.34*width and p0[1] > 0.045*width and p0[2] > -.01*width
  if true then
    vdb.color(c1)
    vdb.triangle(p0,p1,p2)
    vdb.color(c2)
    vdb.triangle(p0,p2,p3)
  end
end



-- set implicit function
--set_sphere_field(grid.vertices)
--set_hyperboloid(grid.vertices)
set_sins(grid.vertices)

-- gen quads
cut_edges(grid.edges)
gather_dual_pos(grid.cells)
-- draw quads
debug_draw_quads(quads)




print('push key')
io.read()









