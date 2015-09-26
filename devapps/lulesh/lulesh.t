import 'compiler.liszt'

--L.default_processor = L.GPU

local PN    = L.require 'lib.pathname'
local Grid  = L.require "domains.grid"


local N = 45
local sz = 1.125
local grid = Grid.NewGrid3d({
  size   = {  N,  N,  N },
  origin = {  0,  0,  0 },
  width  = { sz, sz, sz },

})

local twelfth = 1.0/12.0
local eighth = 1.0/8.0 
------------------------------------------------------------------------------------------
--[[ Element-centered temporaries                                                     ]]--
------------------------------------------------------------------------------------------
-- New relative volume temp
grid.cells:NewField("vnew", L.double):Load(0.0)

grid.cells:NewField("scratchpade01", L.double):Load(0.0)
grid.cells:NewField("scratchpade02", L.double):Load(0.0)
grid.cells:NewField("scratchpade03", L.double):Load(0.0)
grid.cells:NewField("scratchpade04", L.double):Load(0.0)
grid.cells:NewField("scratchpade05", L.double):Load(0.0)
grid.cells:NewField("scratchpade06", L.double):Load(0.0)


------------------------------------------------------------------------------------------
--[[ Node-centered properties                                                         ]]--
------------------------------------------------------------------------------------------
--grid.vertices:NewField("position",  L.vec3d):Load({0.0,0.0,0.0})
--grid.vertices:NewField("velocity",  L.vec3d):Load({0.0,0.0,0.0})
--grid.vertices:NewField("forces",    L.vec3d):Load({0.0,0.0,0.0})
-- Position
grid.vertices:NewField("px", L.double):Load(0)
grid.vertices:NewField("py", L.double):Load(0)
grid.vertices:NewField("pz", L.double):Load(0)
-- Velocity
grid.vertices:NewField("vx", L.double):Load(0)
grid.vertices:NewField("vy", L.double):Load(0)
grid.vertices:NewField("vz", L.double):Load(0)
-- Forces
grid.vertices:NewField("fx", L.double):Load(0)
grid.vertices:NewField("fy", L.double):Load(0)
grid.vertices:NewField("fz", L.double):Load(0)

grid.vertices:NewField("mass",      L.float):Load(0) -- nodal mass

grid.vertices:NewFieldMacro("position", L.NewMacro(function(v) return liszt `{v.px, v.py, v.pz} end))
grid.vertices:NewFieldMacro("velocity", L.NewMacro(function(v) return liszt `{v.vx, v.vy, v.vz} end))
grid.vertices:NewFieldMacro("forces",   L.NewMacro(function(v) return liszt `{v.fx, v.fy, v.fz} end))


------------------------------------------------------------------------------------------
--[[ Element-centered properties                                                      ]]--
------------------------------------------------------------------------------------------
-- All cells are the same material for now.  In the future, cells could be split into
-- multiple materials using grid.cells:NewSetFromFunction()
local material = grid.cells

grid.cells:NewField("e",      L.double)         -- Energy
grid.cells:NewField("p",      L.double):Load(0.0) -- Pressure
grid.cells:NewField("q",      L.double):Load(0.0) -- q
grid.cells:NewField("ql",     L.double):Load(0.0) -- Linear term for q
grid.cells:NewField("qq",     L.double):Load(0.0) -- Quadratic term for q
grid.cells:NewField("vol",    L.double):Load(1.0) -- Relative volume
grid.cells:NewField("volo",   L.double):Load(0.0) -- Reference volume
grid.cells:NewField("delv",   L.double):Load(0.0) -- vnew - v
grid.cells:NewField("vdov",   L.double):Load(0.0) -- Volume derivative over volume
grid.cells:NewField("arealg", L.double):Load(0.0) -- Characteristic length of an element
grid.cells:NewField("ss",     L.double):Load(0.0) -- sound speed
grid.cells:NewField("mass",   L.double):Load(0.0) -- element mass

grid.cells:NewField("force_temp", L.vector(L.double, 3*8))

---- For convenience in matching orientation w/previous lulesh version
-- This is also needed to maintain the code style which allows
-- extensibility to less structured Hex-Mesh topologies
grid.cells:NewField('v', L.vector(grid.vertices, 8))
grid.cells:foreach(liszt( c : grid.cells )
  c.v[0] = c.vertex(0,1,1)
  c.v[1] = c.vertex(1,1,1)
  c.v[2] = c.vertex(1,0,1)
  c.v[3] = c.vertex(0,0,1)
  c.v[4] = c.vertex(0,1,0)
  c.v[5] = c.vertex(1,1,0)
  c.v[6] = c.vertex(1,0,0)
  c.v[7] = c.vertex(0,0,0)
end)

-- Energy is concentrated in the origin node at the beginning of the simulation
grid.cells.e:Load(function(index)
  if index == 0 then return 3.948746e+7
  else return 0
  end
end)

------------------------------------------------------------------------------------------
--[[ Global mesh parameters                                                           ]]--
------------------------------------------------------------------------------------------
local m = {}

m.dtfixed             = -1.0e-7    -- fixed time increment
m.time                = 0          -- current time
m.deltatime           = L.Global(L.float, 0.0)        -- variable time increment
m.deltatimemultlb     = 1.1
m.deltatimemultub     = 1.2
m.stoptime            = 1.0e-2     -- end time for simulation
m.cycle               = 0          -- iteration count for simulation

m.hgcoef              = 3.0        -- hourglass control
m.qstop               = 1.0e12     -- excessive q indicator
m.monoq_max_slope     = 1.0
m.monoq_limiter_mult  = 2.0
m.e_cut               = 1.0e-7     -- energy tolerance
m.p_cut               = 1.0e-7     -- pressure tolerance
m.u_cut               = 1.0e-7     -- velocity tolerance
m.q_cut               = 1.0e-7     -- q tolerance
m.v_cut               = 1.0e-7     -- relative volume tolerance
m.qlc_monoq           = 0.5        -- linear term coef for q
m.qqc_monoq           = 2.0/3.0    -- quadratic term coef for q
m.qqc                 = 2.0
m.eosvmax             = 1.0e9
m.eosvmin             = 1.0e-9
m.pmin                = 0          -- pressure floor
m.emin                = -1.0e15    -- energy floor
m.dvovmax             = 0.1        -- maximum allowable volume change
m.refdens             = 1.0        -- reference density

m.dtcourant           = 0.0        -- courant constraint
m.dtcourant_tmp_max   = 1.0e20
m.dthydro             = 0.0        -- volume change constraint
m.dtmax               = 1.0e-2     -- maximum allowable time increment

m.sizeX = N
m.sizeY = N
m.sizeZ = N

m.timeCourant          = 0.0
m.timeHydro            = 0.0
m.timePosition         = 0.0
m.timeUpdateVol        = 0.0
m.timeIntegrateStress  = 0.0
m.timeHourglass        = 0.0
m.timeKinQ             = 0.0
m.timeQRegionEOS       = 0.0

local start_time = 0
local end_time   = 0

local TimeType = double

-- Global reduction variables for computing time constraints
local dtcourant_tmp = L.Global(L.double, 0.0)
local dthydro_tmp   = L.Global(L.double, 0.0)


-------------------------------------------------------------------------------
--[[ Helper functions and mapped functions                                 ]]--
-------------------------------------------------------------------------------
local liszt fabs (num)
  var result = num
  if num < 0 then result = -num end
  return result
end

-- called over individual cells, returns a 24-vector representing an 8x3 matrix in row-major
-- form
local liszt getLocalNodeCoordVectors (c)
  return {
    c.v[0].px, c.v[0].py, c.v[0].pz,
    c.v[1].px, c.v[1].py, c.v[1].pz,
    c.v[2].px, c.v[2].py, c.v[2].pz,
    c.v[3].px, c.v[3].py, c.v[3].pz,
    c.v[4].px, c.v[4].py, c.v[4].pz,
    c.v[5].px, c.v[5].py, c.v[5].pz,
    c.v[6].px, c.v[6].py, c.v[6].pz,
    c.v[7].px, c.v[7].py, c.v[7].pz
    }
end

local liszt getLocalNodeVelocityVectors (c)
  return {
    c.v[0].vx, c.v[0].vy, c.v[0].vz,
    c.v[1].vx, c.v[1].vy, c.v[1].vz,
    c.v[2].vx, c.v[2].vy, c.v[2].vz,
    c.v[3].vx, c.v[3].vy, c.v[3].vz,
    c.v[4].vx, c.v[4].vy, c.v[4].vz,
    c.v[5].vx, c.v[5].vy, c.v[5].vz,
    c.v[6].vx, c.v[6].vy, c.v[6].vz,
    c.v[7].vx, c.v[7].vy, c.v[7].vz
    }
end
local liszt row (localCoords, i)
  return {localCoords[3*i], localCoords[3*i+1], localCoords[3*i+2]}
end

local liszt TRIPLE_PRODUCT (a,b,c)
  return L.dot(a, L.cross(b,c))
end
local liszt calcElemVolume (c)
  var d61 = c.v[6].position - c.v[1].position
  var d70 = c.v[7].position - c.v[0].position
  var d63 = c.v[6].position - c.v[3].position
  var d20 = c.v[2].position - c.v[0].position

  var d50 = c.v[5].position - c.v[0].position
  var d64 = c.v[6].position - c.v[4].position
  var d31 = c.v[3].position - c.v[1].position
  var d72 = c.v[7].position - c.v[2].position

  var d43 = c.v[4].position - c.v[3].position
  var d57 = c.v[5].position - c.v[7].position
  var d14 = c.v[1].position - c.v[4].position
  var d25 = c.v[2].position - c.v[5].position

  var volume = TRIPLE_PRODUCT( d31+d72, d63, d20 ) +
               TRIPLE_PRODUCT( d43+d57, d64, d70 ) +
               TRIPLE_PRODUCT( d14+d25, d61, d50 )
  return volume * twelfth
end
--local liszt calcElemVolume (localCoords)
--  var d61 = row(localCoords,6) - row(localCoords,1)
--  var d70 = row(localCoords,7) - row(localCoords,0)
--  var d63 = row(localCoords,6) - row(localCoords,3)
--  var d20 = row(localCoords,2) - row(localCoords,0)
--  var d50 = row(localCoords,5) - row(localCoords,0)
--  var d64 = row(localCoords,6) - row(localCoords,4)
--  var d31 = row(localCoords,3) - row(localCoords,1)
--  var d72 = row(localCoords,7) - row(localCoords,2)
--  var d43 = row(localCoords,4) - row(localCoords,3)
--  var d57 = row(localCoords,5) - row(localCoords,7)
--  var d14 = row(localCoords,1) - row(localCoords,4)
--  var d25 = row(localCoords,2) - row(localCoords,5)
--  var volume = L.dot(d31 + d72, L.cross(d63, d20)) +
--               L.dot(d43 + d57, L.cross(d64, d70)) +
--               L.dot(d14 + d25, L.cross(d61, d50))
--  return volume / 12.0
--end

local liszt calcElemShapeFunctionDerivatives1 (localCoords)
  var r0 = row(localCoords, 0)
  var r1 = row(localCoords, 1)
  var r2 = row(localCoords, 2)
  var r3 = row(localCoords, 3)
  var r4 = row(localCoords, 4)
  var r5 = row(localCoords, 5)
  var r6 = row(localCoords, 6)
  var r7 = row(localCoords, 7)

  var r60  =  r6 - r0
  var r53  =  r5 - r3
  var r71  =  r7 - r1
  var r42  =  r4 - r2
  var fjet = r60 - r53 + r71 - r42

  var r6053 = r60 + r53
  var r7142 = r71 + r42
  var fjxi = r6053 - r7142
  var fjze = r6053 + r7142
  var cjet = L.cross(fjze, fjxi)

  return 8.0 * 0.125 * 0.125 * 0.125 * L.dot(fjet,cjet)
end

local liszt calcElemShapeFunctionDerivatives2 (localCoords)
  var r0 = row(localCoords, 0)
  var r1 = row(localCoords, 1)
  var r2 = row(localCoords, 2)
  var r3 = row(localCoords, 3)
  var r4 = row(localCoords, 4)
  var r5 = row(localCoords, 5)
  var r6 = row(localCoords, 6)
  var r7 = row(localCoords, 7)
  
  var r60 = r6 - r0
  var r53 = r5 - r3
  var r71 = r7 - r1
  var r42 = r4 - r2
  var fjet = 0.125 * (r60 - r53 + r71 - r42)

  var r6053 = r60 + r53
  var r7142 = r71 + r42 
  var fjxi = 0.125 * (r6053 - r7142)
  var fjze = 0.125 * (r6053 + r7142)
  
  var cjxi = L.cross(fjet, fjze)
  var cjet = L.cross(fjze, fjxi)
  var cjze = L.cross(fjxi, fjet)

  -- calculate partials :
  -- this need only be done for l = 0,1,2,3 since, by symmetry,
  -- (6,7,4,5) = - (0,1,2,3).

  var temp0 = cjxi + cjet
  var temp1 = cjxi - cjet

  var b0 = - temp0 - cjze
  var b1 =   temp1 - cjze
  var b2 =   temp0 - cjze
  var b3 = - temp1 - cjze
  var b4 = - b2
  var b5 = - b3
  var b6 = - b0
  var b7 = - b1
  
  -- calculate jacobian determinant (volume)
  var volume = 8.0 * L.dot(fjet, cjet)
  -- TODO: HACK HACK HACK!!!
  var b8  = { volume, 0.0, 0.0 }
  return {
    b0[0], b0[1], b0[2],
    b1[0], b1[1], b1[2],
    b2[0], b2[1], b2[2],
    b3[0], b3[1], b3[2],
    b4[0], b4[1], b4[2],
    b5[0], b5[1], b5[2],
    b6[0], b6[1], b6[2],
    b7[0], b7[1], b7[2],
    b8[0], b8[1], b8[2]
  }
end

local liszt sumElemFaceNormal (coords20, coords31, stress)
  var bisect0 = coords20 + coords31
  var bisect1 = coords20 - coords31
  var area = -.0625 * L.cross(bisect0, bisect1)
  area[0] *= stress[0]
  area[1] *= stress[1]
  area[2] *= stress[2]

  return area
end

local liszt calcElemNodeNormals (c, stress)
    var x = {c.v[0].px, c.v[1].px, c.v[2].px, c.v[3].px,
             c.v[4].px, c.v[5].px, c.v[6].px, c.v[7].px }
    var y = {c.v[0].py, c.v[1].py, c.v[2].py, c.v[3].py,
             c.v[4].py, c.v[5].py, c.v[6].py, c.v[7].py }
    var z = {c.v[0].pz, c.v[1].pz, c.v[2].pz, c.v[3].pz,
             c.v[4].pz, c.v[5].pz, c.v[6].pz, c.v[7].pz }


    var r0 = {x[0], y[0], z[0]}
    var r1 = {x[1], y[1], z[1]}
    var r2 = {x[2], y[2], z[2]}
    var r3 = {x[3], y[3], z[3]}
    var r4 = {x[4], y[4], z[4]}
    var r5 = {x[5], y[5], z[5]}
    var r6 = {x[6], y[6], z[6]}
    var r7 = {x[7], y[7], z[7]}

    -- evaluate face one: nodes 0, 1, 2, 3
    var temp0 = sumElemFaceNormal(r2-r0, r3-r1, stress)
  
    -- evaluate face two: nodes 0, 4, 5, 1
    var temp1 = sumElemFaceNormal(r5-r0, r1-r4, stress)
  
    -- evaluate face three: nodes 1, 5, 6, 2
    var temp2 = sumElemFaceNormal(r6-r1, r2-r5, stress)
  
    -- evaluate face four: nodes 2, 6, 7, 3
    var temp3 = sumElemFaceNormal(r7-r2, r3-r6, stress)
  
    -- evaluate face five: nodes 3, 7, 4, 0 */
    var temp4 = sumElemFaceNormal(r4-r3, r0-r7, stress)
  
    -- evaluate face six: nodes 4, 7, 6, 5 */
    var temp5 = sumElemFaceNormal(r6-r4, r5-r7, stress)
  
    var r0 = temp0+temp1+temp4
    var r1 = temp0+temp1+temp2
    var r2 = temp0+temp2+temp3
    var r3 = temp0+temp3+temp4
    var r4 = temp1+temp4+temp5
    var r5 = temp1+temp2+temp5
    var r6 = temp2+temp3+temp5
    var r7 = temp3+temp4+temp5

    return {
      r0[0], r0[1], r0[2], r1[0], r1[1], r1[2],
      r2[0], r2[1], r2[2], r3[0], r3[1], r3[2],
      r4[0], r4[1], r4[2], r5[0], r5[1], r5[2],
      r6[0], r6[1], r6[2], r7[0], r7[1], r7[2]
    }
end
local liszt voluDer (p0,p1,p2,p3,p4,p5)
  var m01 = p0.position + p1.position
  var m12 = p1.position + p2.position
  var m04 = p0.position + p4.position
  var m34 = p3.position + p4.position 
  var m25 = p2.position + p5.position
  var m35 = p3.position + p5.position
  
  var dvdx =   m12[1] * m01[2] - m01[1] * m12[2] +
               m04[1] * m34[2] - m34[1] * m04[2] -
               m25[1] * m35[2] + m35[1] * m25[2]

  var dvdy = - m12[0] * m01[2] + m01[0] * m12[2] -
               m04[0] * m34[2] + m34[0] * m04[2] +
               m25[0] * m35[2] - m35[0] * m25[2]

  var dvdz = - m12[1] * m01[0] + m01[1] * m12[0] -
               m04[1] * m34[0] + m34[1] * m04[0] +
               m25[1] * m35[0] - m35[1] * m25[0]

  return { dvdx*twelfth, dvdy*twelfth, dvdz*twelfth }
end

local liszt calcElemVolumeDerivative (c)
  var p0 = c.v[0]
  var p1 = c.v[1]
  var p2 = c.v[2]
  var p3 = c.v[3]
  var p4 = c.v[4]
  var p5 = c.v[5]
  var p6 = c.v[6]
  var p7 = c.v[7]

  var v0 = voluDer(p1, p2, p3, p4, p5, p7)
  var v3 = voluDer(p0, p1, p2, p7, p4, p6)
  var v2 = voluDer(p3, p0, p1, p6, p7, p5)
  var v1 = voluDer(p2, p3, p0, p5, p6, p4)
  var v4 = voluDer(p7, p6, p5, p0, p3, p1)
  var v5 = voluDer(p4, p7, p6, p1, p0, p2)
  var v6 = voluDer(p5, p4, p7, p2, p1, p3)
  var v7 = voluDer(p6, p5, p4, p3, p2, p0)

  return { v0[0], v1[0], v2[0], v3[0], v4[0], v5[0], v6[0], v7[0],
           v0[1], v1[1], v2[1], v3[1], v4[1], v5[1], v6[1], v7[1],
           v0[2], v1[2], v2[2], v3[2], v4[2], v5[2], v6[2], v7[2] }
end

local liszt initialVolumeCalc(c : grid.cells)

  --var localCoords = getLocalNodeCoordVectors(c)
  var volume = calcElemVolume(c)

  c.volo = volume
  c.mass = volume

  var dvol = L.float(volume * eighth)
  c.v[0].mass += dvol
  c.v[1].mass += dvol
  c.v[2].mass += dvol
  c.v[3].mass += dvol
  c.v[4].mass += dvol
  c.v[5].mass += dvol
  c.v[6].mass += dvol
  c.v[7].mass += dvol
end

-- Grid initializes position over cell centers, so we need to
-- initialize vertex positions manually.

local Nth = 1.0 / N

local liszt initVectorPosition(v : grid.vertices)
  v.px = L.double(L.xid(v)) * sz * Nth
  v.py = L.double(L.yid(v)) * sz * Nth
  v.pz = L.double(L.zid(v)) * sz * Nth
end

function m.initMeshParameters ()
  m.deltatime:set(1.0e-7)
  m.time      = 0
  m.cycle     = 0

  m.dtcourant = 1e20
  m.dthydro   = 1e20

  grid.vertices:foreach(initVectorPosition)
  grid.cells:foreach(initialVolumeCalc)
end

function timeIncrement( )
  local targetdt = m.stoptime - m.time

  if m.dtfixed <= 0.0 and m.cycle ~= 0 then
    local olddt = m.deltatime:get()
    local newdt = 1e20

    if m.dtcourant < newdt then newdt = m.dtcourant / 2.0       end
    if m.dthydro   < newdt then newdt = m.dthydro   * 2.0 / 3.0 end

    local ratio = newdt / olddt
    if ratio >= 1.0 then
      if     ratio < m.deltatimemultlb then newdt = olddt
      elseif ratio > m.deltatimemultub then newdt = olddt*m.deltatimemultub
      end
    end

    if newdt > m.dtmax then newdt = m.dtmax end
    m.deltatime:set(newdt)
  end

  local dttmp = m.deltatime:get()

  -- TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE
  if targetdt > dttmp and targetdt < 4.0 * dttmp / 3.0 then
    targetdt = 2.0 * dttmp / 3.0
  end

  if targetdt < dttmp then m.deltatime:set(targetdt) end

  -- Increment
  m.time  = m.time  + m.deltatime:get()
  m.cycle = m.cycle + 1
end

local liszt accumulateForces(c, f)
  c.v[0].fx += row(f,0)[0]
  c.v[0].fy += row(f,0)[1]
  c.v[0].fz += row(f,0)[2]
  c.v[1].fx += row(f,1)[0]
  c.v[1].fy += row(f,1)[1]
  c.v[1].fz += row(f,1)[2]
  c.v[2].fx += row(f,2)[0]
  c.v[2].fy += row(f,2)[1]
  c.v[2].fz += row(f,2)[2]
  c.v[3].fx += row(f,3)[0]
  c.v[3].fy += row(f,3)[1]
  c.v[3].fz += row(f,3)[2]
  c.v[4].fx += row(f,4)[0]
  c.v[4].fy += row(f,4)[1]
  c.v[4].fz += row(f,4)[2]
  c.v[5].fx += row(f,5)[0]
  c.v[5].fy += row(f,5)[1]
  c.v[5].fz += row(f,5)[2]
  c.v[6].fx += row(f,6)[0]
  c.v[6].fy += row(f,6)[1]
  c.v[6].fz += row(f,6)[2]
  c.v[7].fx += row(f,7)[0]
  c.v[7].fy += row(f,7)[1]
  c.v[7].fz += row(f,7)[2]
end


local liszt mmult_4x8x3(ma, mb)
  var result : L.vector(L.double, 12)
  result[0]  = ma[0]*mb[0]  + ma[1]*mb[3]  + ma[2]*mb[6]  + ma[3]*mb[9]   + ma[4]*mb[12]  + ma[5]*mb[15]  + ma[6]*mb[18]  + ma[7]*mb[21]
  result[1]  = ma[0]*mb[1]  + ma[1]*mb[4]  + ma[2]*mb[7]  + ma[3]*mb[10]  + ma[4]*mb[13]  + ma[5]*mb[16]  + ma[6]*mb[19]  + ma[7]*mb[22]
  result[2]  = ma[0]*mb[2]  + ma[1]*mb[5]  + ma[2]*mb[8]  + ma[3]*mb[11]  + ma[4]*mb[14]  + ma[5]*mb[17]  + ma[6]*mb[20]  + ma[7]*mb[23]
  result[3]  = ma[8]*mb[0]  + ma[9]*mb[3]  + ma[10]*mb[6] + ma[11]*mb[9]  + ma[12]*mb[12] + ma[13]*mb[15] + ma[14]*mb[18] + ma[15]*mb[21]
  result[4]  = ma[8]*mb[1]  + ma[9]*mb[4]  + ma[10]*mb[7] + ma[11]*mb[10] + ma[12]*mb[13] + ma[13]*mb[16] + ma[14]*mb[19] + ma[15]*mb[22]
  result[5]  = ma[8]*mb[2]  + ma[9]*mb[5]  + ma[10]*mb[8] + ma[11]*mb[11] + ma[12]*mb[14] + ma[13]*mb[17] + ma[14]*mb[20] + ma[15]*mb[23]
  result[6]  = ma[16]*mb[0] + ma[17]*mb[3] + ma[18]*mb[6] + ma[19]*mb[9]  + ma[20]*mb[12] + ma[21]*mb[15] + ma[22]*mb[18] + ma[23]*mb[21]
  result[7]  = ma[16]*mb[1] + ma[17]*mb[4] + ma[18]*mb[7] + ma[19]*mb[10] + ma[20]*mb[13] + ma[21]*mb[16] + ma[22]*mb[19] + ma[23]*mb[22]
  result[8]  = ma[16]*mb[2] + ma[17]*mb[5] + ma[18]*mb[8] + ma[19]*mb[11] + ma[20]*mb[14] + ma[21]*mb[17] + ma[22]*mb[20] + ma[23]*mb[23]
  result[9]  = ma[24]*mb[0] + ma[25]*mb[3] + ma[26]*mb[6] + ma[27]*mb[9]  + ma[28]*mb[12] + ma[29]*mb[15] + ma[30]*mb[18] + ma[31]*mb[21]
  result[10] = ma[24]*mb[1] + ma[25]*mb[4] + ma[26]*mb[7] + ma[27]*mb[10] + ma[28]*mb[13] + ma[29]*mb[16] + ma[30]*mb[19] + ma[31]*mb[22]
  result[11] = ma[24]*mb[2] + ma[25]*mb[5] + ma[26]*mb[8] + ma[27]*mb[11] + ma[28]*mb[14] + ma[29]*mb[17] + ma[30]*mb[20] + ma[31]*mb[23]
  return result
end

local liszt mmult_4x3x8 (ma, mb)
  var result : L.vector(L.double, 32)
  result[0]  = ma[0]*mb[0] + ma[1]*mb[8]   + ma[2]*mb[16]
  result[1]  = ma[0]*mb[1] + ma[1]*mb[9]   + ma[2]*mb[17]
  result[2]  = ma[0]*mb[2] + ma[1]*mb[10]  + ma[2]*mb[18]
  result[3]  = ma[0]*mb[3] + ma[1]*mb[11]  + ma[2]*mb[19]
  result[4]  = ma[0]*mb[4] + ma[1]*mb[12]  + ma[2]*mb[20]
  result[5]  = ma[0]*mb[5] + ma[1]*mb[13]  + ma[2]*mb[21]
  result[6]  = ma[0]*mb[6] + ma[1]*mb[14]  + ma[2]*mb[22]
  result[7]  = ma[0]*mb[7] + ma[1]*mb[15]  + ma[2]*mb[23]
  result[8]  = ma[3]*mb[0] + ma[4]*mb[8]   + ma[5]*mb[16]
  result[9]  = ma[3]*mb[1] + ma[4]*mb[9]   + ma[5]*mb[17]
  result[10] = ma[3]*mb[2] + ma[4]*mb[10]  + ma[5]*mb[18]
  result[11] = ma[3]*mb[3] + ma[4]*mb[11]  + ma[5]*mb[19]
  result[12] = ma[3]*mb[4] + ma[4]*mb[12]  + ma[5]*mb[20]
  result[13] = ma[3]*mb[5] + ma[4]*mb[13]  + ma[5]*mb[21]
  result[14] = ma[3]*mb[6] + ma[4]*mb[14]  + ma[5]*mb[22]
  result[15] = ma[3]*mb[7] + ma[4]*mb[15]  + ma[5]*mb[23]
  result[16] = ma[6]*mb[0] + ma[7]*mb[8]   + ma[8]*mb[16]
  result[17] = ma[6]*mb[1] + ma[7]*mb[9]   + ma[8]*mb[17]
  result[18] = ma[6]*mb[2] + ma[7]*mb[10]  + ma[8]*mb[18]
  result[19] = ma[6]*mb[3] + ma[7]*mb[11]  + ma[8]*mb[19]
  result[20] = ma[6]*mb[4] + ma[7]*mb[12]  + ma[8]*mb[20]
  result[21] = ma[6]*mb[5] + ma[7]*mb[13]  + ma[8]*mb[21]
  result[22] = ma[6]*mb[6] + ma[7]*mb[14]  + ma[8]*mb[22]
  result[23] = ma[6]*mb[7] + ma[7]*mb[15]  + ma[8]*mb[23]
  result[24] = ma[9]*mb[0] + ma[10]*mb[8]  + ma[11]*mb[16]
  result[25] = ma[9]*mb[1] + ma[10]*mb[9]  + ma[11]*mb[17]
  result[26] = ma[9]*mb[2] + ma[10]*mb[10] + ma[11]*mb[18]
  result[27] = ma[9]*mb[3] + ma[10]*mb[11] + ma[11]*mb[19]
  result[28] = ma[9]*mb[4] + ma[10]*mb[12] + ma[11]*mb[20]
  result[29] = ma[9]*mb[5] + ma[10]*mb[13] + ma[11]*mb[21]
  result[30] = ma[9]*mb[6] + ma[10]*mb[14] + ma[11]*mb[22]
  result[31] = ma[9]*mb[7] + ma[10]*mb[15] + ma[11]*mb[23]
  return result
end

local liszt mmult_8x4x3 (ma, mb)
  var result : L.vector(L.double, 24)
  result[0]  = ma[0]*mb[0]  + ma[1]*mb[3]  + ma[2]*mb[6]  + ma[3]*mb[9]
  result[1]  = ma[0]*mb[1]  + ma[1]*mb[4]  + ma[2]*mb[7]  + ma[3]*mb[10]
  result[2]  = ma[0]*mb[2]  + ma[1]*mb[5]  + ma[2]*mb[8]  + ma[3]*mb[11]
  result[3]  = ma[4]*mb[0]  + ma[5]*mb[3]  + ma[6]*mb[6]  + ma[7]*mb[9]
  result[4]  = ma[4]*mb[1]  + ma[5]*mb[4]  + ma[6]*mb[7]  + ma[7]*mb[10]
  result[5]  = ma[4]*mb[2]  + ma[5]*mb[5]  + ma[6]*mb[8]  + ma[7]*mb[11]
  result[6]  = ma[8]*mb[0]  + ma[9]*mb[3]  + ma[10]*mb[6] + ma[11]*mb[9]
  result[7]  = ma[8]*mb[1]  + ma[9]*mb[4]  + ma[10]*mb[7] + ma[11]*mb[10]
  result[8]  = ma[8]*mb[2]  + ma[9]*mb[5]  + ma[10]*mb[8] + ma[11]*mb[11]
  result[9]  = ma[12]*mb[0] + ma[13]*mb[3] + ma[14]*mb[6] + ma[15]*mb[9]
  result[10] = ma[12]*mb[1] + ma[13]*mb[4] + ma[14]*mb[7] + ma[15]*mb[10]
  result[11] = ma[12]*mb[2] + ma[13]*mb[5] + ma[14]*mb[8] + ma[15]*mb[11]
  result[12] = ma[16]*mb[0] + ma[17]*mb[3] + ma[18]*mb[6] + ma[19]*mb[9]
  result[13] = ma[16]*mb[1] + ma[17]*mb[4] + ma[18]*mb[7] + ma[19]*mb[10]
  result[14] = ma[16]*mb[2] + ma[17]*mb[5] + ma[18]*mb[8] + ma[19]*mb[11]
  result[15] = ma[20]*mb[0] + ma[21]*mb[3] + ma[22]*mb[6] + ma[23]*mb[9]
  result[16] = ma[20]*mb[1] + ma[21]*mb[4] + ma[22]*mb[7] + ma[23]*mb[10]
  result[17] = ma[20]*mb[2] + ma[21]*mb[5] + ma[22]*mb[8] + ma[23]*mb[11]
  result[18] = ma[24]*mb[0] + ma[25]*mb[3] + ma[26]*mb[6] + ma[27]*mb[9]
  result[19] = ma[24]*mb[1] + ma[25]*mb[4] + ma[26]*mb[7] + ma[27]*mb[10]
  result[20] = ma[24]*mb[2] + ma[25]*mb[5] + ma[26]*mb[8] + ma[27]*mb[11]
  result[21] = ma[28]*mb[0] + ma[29]*mb[3] + ma[30]*mb[6] + ma[31]*mb[9]
  result[22] = ma[28]*mb[1] + ma[29]*mb[4] + ma[30]*mb[7] + ma[31]*mb[10]
  result[23] = ma[28]*mb[2] + ma[29]*mb[5] + ma[30]*mb[8] + ma[31]*mb[11]
  return result
end

local transpose_4x8 = L.NewMacro(function(m)
  return liszt `{
    m[0],  m[8], m[16], m[24],
    m[1],  m[9], m[17], m[25],
    m[2], m[10], m[18], m[26],
    m[3], m[11], m[19], m[27],
    m[4], m[12], m[20], m[28],
    m[5], m[13], m[21], m[29],
    m[6], m[14], m[22], m[30],
    m[7], m[15], m[23], m[31]
  }
end)



local liszt calcVolumeForceForElems (c : grid.cells)
  var determ = c.volo * c.vol
  var volinv = 1.0 / determ
  var volume13 = L.cbrt(determ)
  var coefficient = -m.hgcoef * 0.01 * c.ss * c.mass / volume13
  var localCoords = getLocalNodeCoordVectors(c)
  var pf = calcElemVolumeDerivative(c)

  -- 4x8 matrix
  var gamma : L.vector(L.double, 32) = {
     1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0,
     1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,
     1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0
  }

  -- 4x8 matrix
  var hourgam = gamma - volinv *
                        mmult_4x3x8(mmult_4x8x3(gamma, localCoords), pf) -- this is CalcHourglassModes
  -- 8x4 matrix
  var hourgamXpose = transpose_4x8(hourgam)
  -- 8x3 matrix
  var localVelocities = getLocalNodeVelocityVectors(c)
  var hgf = coefficient *
            mmult_8x4x3(hourgamXpose, mmult_4x8x3(hourgam, localVelocities))
  -- Volume calculation involves extra work for numerical consistency.
  if m.hgcoef > 0.0 then
    var stress = -c.p - c.q
    var f = calcElemNodeNormals(c, {stress, stress, stress})
    hgf += f
  --accumulateForces(c, f)
  end
  accumulateForces(c, hgf)
  --c.force_temp = hgf
end

local liszt calcPositionForNodes(v : grid.vertices)
  var accel = v.forces / v.mass
  -- Enforce boundary conditions of symmetry planes
  if L.xid(v) == 0 then accel[0] = 0 end
  if L.yid(v) == 0 then accel[1] = 0 end
  if L.zid(v) == 0 then accel[2] = 0 end

  var vtmp = v.velocity + accel * m.deltatime

  if fabs(vtmp[0]) < m.u_cut then vtmp[0] = 0.0 end
  if fabs(vtmp[1]) < m.u_cut then vtmp[1] = 0.0 end
  if fabs(vtmp[2]) < m.u_cut then vtmp[2] = 0.0 end
  v.vx = vtmp[0]
  v.vy = vtmp[1]
  v.vz = vtmp[2]

  var pdelta = vtmp * m.deltatime
  v.px += pdelta[0]
  v.py += pdelta[1]
  v.pz += pdelta[2]
  
  v.fx = 0
  v.fy = 0
  v.fz = 0
end

function lagrangeNodal ()
  grid.cells:foreach(calcVolumeForceForElems)
  grid.vertices:foreach(calcPositionForNodes)
end

local liszt calcElemCharacteristicLength(localCoords, volume)
  var r0 = row(localCoords, 0)
  var r1 = row(localCoords, 1)
  var r2 = row(localCoords, 2)
  var r3 = row(localCoords, 3)
  var r4 = row(localCoords, 4)
  var r5 = row(localCoords, 5)
  var r6 = row(localCoords, 6)
  var r7 = row(localCoords, 7)

  var f : L.vec3d = {0.0, 0.0, 0.0}
  var g : L.vec3d = {0.0, 0.0, 0.0}
  var temp       : L.double = 0.0
  var a          : L.double = 0.0
  var charlength : L.double = 0.0

  f = (r2 - r0) - (r3 - r1)
  g = (r2 - r0) + (r3 - r1)
  temp = L.dot(f,g)
  a = L.dot(f,f)*L.dot(g,g) - temp*temp
  charlength max= a

  f = (r6 - r4) - (r7 - r5)
  g = (r6 - r4) + (r7 - r5)
  temp = L.dot(f,g)
  a = L.dot(f,f)*L.dot(g,g) - temp*temp
  charlength max= a

  f = (r5 - r0) - (r4 - r1)
  g = (r5 - r0) + (r4 - r1)
  temp = L.dot(f,g)
  a = L.dot(f,f)*L.dot(g,g) - temp*temp
  charlength max= a

  f = (r6 - r1) - (r5 - r2)
  g = (r6 - r1) + (r5 - r2)
  temp = L.dot(f,g)
  a = L.dot(f,f)*L.dot(g,g) - temp*temp
  charlength max= a

  f = (r7 - r2) - (r6 - r3)
  g = (r7 - r2) + (r6 - r3)
  temp = L.dot(f,g)
  a = L.dot(f,f)*L.dot(g,g) - temp*temp
  charlength max= a

  f = (r4 - r3) - (r7 - r0)
  g = (r4 - r3) + (r7 - r0)
  temp = L.dot(f,g)
  a = L.dot(f,f)*L.dot(g,g) - temp*temp
  charlength max= a

  charlength = 4.0 * volume / L.sqrt(charlength)

  return charlength
end

local liszt calcElemVelocityGradient(localVelocities, pf, detJ)
  var inv_detJ = 1.0 / detJ
  var r06 = row(localVelocities,0) - row(localVelocities,6)
  var r17 = row(localVelocities,1) - row(localVelocities,7)
  var r24 = row(localVelocities,2) - row(localVelocities,4)
  var r35 = row(localVelocities,3) - row(localVelocities,5)

  var pfr0 = row(pf,0)
  var pfr1 = row(pf,1)
  var pfr2 = row(pf,2)
  var pfr3 = row(pf,3)

  var temp0 = inv_detJ * ((pfr0[0]*r06[0]) + pfr1[0]*r17[0] + pfr2[0]*r24[0] + pfr3[0]*r35[0])
  var temp1 = inv_detJ * ((pfr0[1]*r06[1]) + pfr1[1]*r17[1] + pfr2[1]*r24[1] + pfr3[1]*r35[1])
  var temp2 = inv_detJ * ((pfr0[2]*r06[2]) + pfr1[2]*r17[2] + pfr2[2]*r24[2] + pfr3[2]*r35[2])

  var pfc0 = {pfr0[0], pfr1[0], pfr2[0], pfr3[0]}
  var pfc1 = {pfr0[1], pfr1[1], pfr2[1], pfr3[1]}
  var pfc2 = {pfr0[2], pfr1[2], pfr2[2], pfr3[2]}

  var dyddx = inv_detJ * (pfc0[0] * r06[1]
                       +  pfc0[1] * r17[1]
                       +  pfc0[2] * r24[1]
                       +  pfc0[3] * r35[1])

  var dxddy = inv_detJ * (pfc1[0] * r06[0]
                       +  pfc1[1] * r17[0]
                       +  pfc1[2] * r24[0]
                       +  pfc1[3] * r35[0])

  var dzddx = inv_detJ * (pfc0[0] * r06[2]
                       +  pfc0[1] * r17[2]
                       +  pfc0[2] * r24[2]
                       +  pfc0[3] * r35[2])

  var dxddz = inv_detJ * (pfc2[0] * r06[0]
                       +  pfc2[1] * r17[0]
                       +  pfc2[2] * r24[0]
                       +  pfc2[3] * r35[0])

  var dzddy = inv_detJ * (pfc1[0] * r06[2]
                       +  pfc1[1] * r17[2]
                       +  pfc1[2] * r24[2]
                       +  pfc1[3] * r35[2])

  var dyddz = inv_detJ * (pfc2[0] * r06[1]
                       +  pfc2[1] * r17[1]
                       +  pfc2[2] * r24[1]
                       +  pfc2[3] * r35[1])

  return {temp0, temp1, temp2, 0.5*(dzddy+dyddz), 0.5*(dxddz+dzddx), 0.5*(dxddy+dyddx)}
end

-- This function uses the scratchpade01-03 fields to temporarily store 2nd derivatives
-- of position (?) in x, y, z
local third = 1.0/3.0
local liszt calcKinematicsForElem(c, localCoords, localVelocities)
  var volume = calcElemVolume(c)
  var relativeVolume = volume / c.volo
  c.vnew = relativeVolume
  -- Volume error detection
  L.assert(c.vnew > 0.0)
  c.delv = relativeVolume - c.vol

  c.arealg = calcElemCharacteristicLength(localCoords, volume)

  var dt2 = 0.5 * m.deltatime
  for i = 0, 24, 4 do
    localCoords[i] -= dt2 * localVelocities[i]
    localCoords[i+1] -= dt2 * localVelocities[i+1]
    localCoords[i+2] -= dt2 * localVelocities[i+2]
    localCoords[i+3] -= dt2 * localVelocities[i+3]
  end

  var b = calcElemShapeFunctionDerivatives2(localCoords)
  var detJ = row(b,8)[0]

  var D = calcElemVelocityGradient(localVelocities, b, detJ)
  var vdov_tmp = D[0] + D[1] + D[2]
  var vdovthird = vdov_tmp * third
  c.vdov = vdov_tmp
  c.scratchpade01 = D[0] - vdovthird -- dxx
  c.scratchpade02 = D[1] - vdovthird -- dyy
  c.scratchpade03 = D[2] - vdovthird -- dzz
end

-- TODO
local liszt calcMonotonicQGradientsForElem(c, localCoords, localVelocities)
  var ptiny = 1.e-36

  var rc0 = row(localCoords,7)
  var rc1 = row(localCoords,6)
  var rc2 = row(localCoords,5)
  var rc3 = row(localCoords,4)
  var rc4 = row(localCoords,3)
  var rc5 = row(localCoords,2)
  var rc6 = row(localCoords,1)
  var rc7 = row(localCoords,0)

  var rv0 = row(localVelocities,7)
  var rv1 = row(localVelocities,6)
  var rv2 = row(localVelocities,5)
  var rv3 = row(localVelocities,4)
  var rv4 = row(localVelocities,3)
  var rv5 = row(localVelocities,2)
  var rv6 = row(localVelocities,1)
  var rv7 = row(localVelocities,0)

  var vol  = c.volo*c.vnew
  var norm = 1.0 / (vol + ptiny) 

  var dj = -0.25 * ((rc0 + rc1 + rc5 + rc4) - (rc3 + rc2 + rc6 + rc7))
  var di =  0.25 * ((rc1 + rc2 + rc6 + rc5) - (rc0 + rc3 + rc7 + rc4))
  var dk =  0.25 * ((rc4 + rc5 + rc6 + rc7) - (rc0 + rc1 + rc2 + rc3))

  --[[ velocity gradient delv_xi, delv_eta, delv_zeta
       stored in scratchpade01-03
     position gradient delx_xi, delx_eta, delx_zeta
         stored in scratchpade04-0
  ]]--
  var a  : L.vec3d
  var dv : L.vec3d

  -- find delvk and delxk ( i cross j )
  a = L.cross(di,dj)
  c.scratchpade06 = vol / L.sqrt(L.dot(a,a) + ptiny)
  a = norm * a
  dv = 0.25 * ((rv4 + rv5 + rv6 + rv7) - (rv0 + rv1 + rv2 + rv3))
  c.scratchpade03 = L.dot(a, dv)

  -- find delxi and delvi ( j cross k )
  a = L.cross(dj,dk)
  c.scratchpade04 = vol / L.sqrt(L.dot(a,a) + ptiny)
  a = norm * a
  dv = 0.25 * ((rv1 + rv2 + rv6 + rv5) - (rv0 + rv3 + rv7 + rv4))
  c.scratchpade01 = L.dot(a, dv)

  -- find delxj and delvj ( k cross i )
  a = L.cross(dk,di)
  c.scratchpade05 = vol / L.sqrt(L.dot(a,a) + ptiny)
  a = norm * a
  dv = -0.25 * ((rv0 + rv1 + rv5 + rv4) - (rv3 + rv2 + rv6 + rv7))
  c.scratchpade02 = L.dot(a, dv)

end

-- fused Kinematics and Monotonic Q Gradient calculations
local liszt calcKinemAndMQGradientsForElems1(c : grid.cells)
  var localCoords     = getLocalNodeCoordVectors(c)
  var localVelocities = getLocalNodeVelocityVectors(c)
  calcKinematicsForElem(c, localCoords, localVelocities)
  end
local liszt calcKinemAndMQGradientsForElems2(c : grid.cells)
  var localCoords     = getLocalNodeCoordVectors(c)
  var localVelocities = getLocalNodeVelocityVectors(c)
  calcMonotonicQGradientsForElem(c, localCoords, localVelocities)
end

-- TODO
local liszt calcMonotonicQRegionForElem(c)
    --[[
      velocity gradient:
        delv_xi    => scratchpade01
        delv_eta   => scratchpade02
        delv_zeta  => scratchpade03

      position gradient
        delx_xi    => scratchpade04
        delx_eta   => scratchpade05
        delx_zeta  => scratchpade06
    ]]--

  var ptiny : L.double = 1.e-36
  
  var qlin    : L.double = 0.0
  var qquad   : L.double = 0.0
  var phixi   : L.double = 0.0
  var phieta  : L.double = 0.0
  var phizeta : L.double = 0.0

  var norm : L.double = 0.0

  var delvm : L.double = 0.0
  var delvp : L.double = 0.0

  var delv_xi_tmp   = c.scratchpade01
  var delv_eta_tmp  = c.scratchpade02
  var delv_zeta_tmp = c.scratchpade03


  var monoq_limiter_mult_tmp = m.monoq_limiter_mult
  var monoq_max_slope_tmp    = m.monoq_max_slope

  --[[ ** phixi ** ]]--
  norm = 1.0 / (delv_xi_tmp + ptiny)

  if L.xid(c) == 0 then -- On X-symmetry plane
    delvm = delv_xi_tmp
  else
    delvm = c(-1,0,0).scratchpade01
  end

  if L.xid(c) == (N-1) then -- On X-free plane
    delvp = 0.0
  else
    delvp = c(1,0,0).scratchpade01
  end

  delvm *= norm
  delvp *= norm

  phixi = 0.5 * (delvm + delvp)
  delvm *= monoq_limiter_mult_tmp
  delvp *= monoq_limiter_mult_tmp

  if delvm < phixi               then phixi = delvm               end
  if delvp < phixi               then phixi = delvp               end
  if phixi < 0.0                 then phixi = 0.0                 end
  if phixi > monoq_max_slope_tmp then phixi = monoq_max_slope_tmp end

  --[[ ** phieta ** ]]--
  norm = 1.0 / (delv_eta_tmp + ptiny)

  if L.yid(c) == 0 then -- On Y-symmetry plane
    delvm = delv_eta_tmp
  else
    delvm = c(0,-1,0).scratchpade02
  end

  if L.yid(c) == N - 1 then -- On Y-free plane
    delvp = 0.0
  else
    delvp = c(0,1,0).scratchpade02
  end

  delvm *= norm 
  delvp *= norm 

  phieta = 0.5 * (delvm + delvp) 
  delvm *= monoq_limiter_mult_tmp
  delvp *= monoq_limiter_mult_tmp

  if delvm  < phieta              then phieta = delvm               end
  if delvp  < phieta              then phieta = delvp               end
  if phieta < 0.0                 then phieta = 0.0                 end
  if phieta > monoq_max_slope_tmp then phieta = monoq_max_slope_tmp end

  --[[ ** phizeta ** ]]--
  norm = 1.0 / (delv_zeta_tmp + ptiny)

  if L.zid(c) == 0 then
    delvm = delv_zeta_tmp
  else
    delvm = c(0,0,-1).scratchpade03
  end

  if L.zid(c) == N - 1 then
    delvp = 0.0
  else
    delvp = c(0,0,1).scratchpade03
  end

  delvm *= norm 
  delvp *= norm 

  phizeta = 0.5 * (delvm + delvp) 
  delvm *= monoq_limiter_mult_tmp
  delvp *= monoq_limiter_mult_tmp

  if delvm   < phizeta             then phizeta = delvm               end
  if delvp   < phizeta             then phizeta = delvp               end
  if phizeta < 0.0                 then phizeta = 0.0                 end
  if phizeta > monoq_max_slope_tmp then phizeta = monoq_max_slope_tmp end

  -- Remove length scale
  if c.vdov > 0.0 then
    qlin  = 0.0
    qquad = 0.0
  else
    var delvxxi   = delv_xi_tmp   * c.scratchpade04
    var delvxeta  = delv_eta_tmp  * c.scratchpade05
    var delvxzeta = delv_zeta_tmp * c.scratchpade06
    if delvxxi   > 0.0 then delvxxi   = 0.0 end
    if delvxeta  > 0.0 then delvxeta  = 0.0 end
    if delvxzeta > 0.0 then delvxzeta = 0.0 end

    var rho = c.mass / (c.volo * c.vnew)
    qlin = -m.qlc_monoq * rho * (delvxxi * (1.0 - phixi) + delvxeta * (1.0 - phieta) + delvxzeta * (1.0 - phizeta))
    qquad = m.qqc_monoq * rho * (delvxxi*delvxxi * (1.0 - phixi*phixi) + delvxeta*delvxeta * (1.0 - phieta*phieta) + delvxzeta*delvxzeta * (1.0 - phizeta*phizeta))
  end

  c.qq = qquad
  c.ql = qlin

  L.assert(qlin <= m.qstop)
end

-- TODO
local liszt calcSoundSpeedForElem(vnewc, rho0, enewc, pnewc, pbvc, bvc)
  var ss = (pbvc * enewc + vnewc * vnewc * bvc * pnewc) / rho0
  if ss <= 0.111111e-36 then
    ss = 0.3333333e-18
  else 
    ss = L.sqrt(ss)
  end
  return ss
end

local two_thirds = 2./3.
local liszt calcPressureForElems(e_old, compression, vnewc, pmin, p_cut, eosvmax)
  var c1s   : L.double = two_thirds
  var bvc   : L.double = c1s * (compression + 1)
  var pbvc  : L.double = c1s
  var p_new : L.double = bvc * e_old

  if fabs(p_new) <  p_cut   then p_new = 0.0 end
  if vnewc       >= eosvmax then p_new = 0.0 end
  p_new max= pmin

  return { p_new, bvc, pbvc }
end

local sixth = 1./6.
local liszt calcEnergyForElems (p_old, e_old, q_old, compression, compHalfStep, vnewc, 
                                         work, delvc, qq_old, ql_old, eosvmax, rho0)
  var emin_tmp = m.emin
  var e_new : L.double = m.emin
  e_new max= (e_old - 0.5 * delvc * (p_old + q_old) + 0.5 * work)

  var p_cut_tmp = m.p_cut
  var pmin_tmp  = m.pmin
  var retVal    = calcPressureForElems(e_new, compHalfStep, vnewc, pmin_tmp, p_cut_tmp, eosvmax)
  var pHalfStep = retVal[0]
  var bvc       = retVal[1]
  var pbvc      = retVal[2]

  var vhalf = 1.0 / (1.0 + compHalfStep)
  var ssc = calcSoundSpeedForElem(vhalf, rho0, e_new, pHalfStep, pbvc, bvc)

  var q_new : L.double = 0.0
  if delvc <= 0.0 then q_new = ssc*ql_old + qq_old end

  e_new += 0.5 * delvc * (3.0 * (p_old + q_old) - 4.0 * (pHalfStep + q_new)) + (0.5 * work)

  var e_cut_tmp = m.e_cut

  if fabs(e_new) < e_cut_tmp then e_new = 0.0 end
  e_new max= emin_tmp

  retVal = calcPressureForElems(e_new, compression, vnewc, pmin_tmp, p_cut_tmp, eosvmax)
  var p_new = retVal[0]
  bvc       = retVal[1]
  pbvc      = retVal[2]
  ssc = calcSoundSpeedForElem(vnewc, rho0, e_new, p_new, pbvc, bvc)
  var q_tilde : L.double = 0.0
  if delvc <= 0.0 then q_tilde = ssc*ql_old + qq_old end
  e_new -= (7.0*(p_old + q_old) - 8.0*(pHalfStep + q_new) + (p_new + q_tilde)) * delvc * sixth
  if fabs(e_new) < e_cut_tmp then e_new = 0.0 end
  e_new max= emin_tmp

  retVal = calcPressureForElems(e_new, compression, vnewc, pmin_tmp, p_cut_tmp, eosvmax)
  p_new     = retVal[0]
  bvc       = retVal[1]
  pbvc      = retVal[2]
  if delvc <= 0.0 then
    ssc = calcSoundSpeedForElem(vnewc, rho0, e_new, p_new, pbvc, bvc)
    q_new = ssc*ql_old + qq_old
    if fabs(q_new) < m.q_cut then q_new = 0.0 end
  end
  return { p_new, e_new, q_new, bvc, pbvc }
end

local liszt updateVolumesForElem (c)
  var tmpV = c.vnew
  if fabs(tmpV - 1.0) < m.v_cut then tmpV = 1.0 end
  c.vol = tmpV
end

local liszt evalEOSForElem (c)
  var rho0         = m.refdens
  var vnewc        = c.vnew
  var delvc        = c.delv
  var e_old        = c.e
  var p_old        = c.p
  var q_old        = c.q
  var qqtmp        = c.qq
  var qltmp        = c.ql
  var work         = 0
  var compression  = 1 / vnewc - 1
  var vchalf       = vnewc - delvc * 0.5
  var compHalfStep = 1 / vchalf - 1

  var eosvmax_tmp = m.eosvmax
  var eosvmin_tmp = m.eosvmin

  -- Check for v > m.eosvmax or v < m.eosvmin
  if eosvmin_tmp ~= 0 and vnewc < eosvmin_tmp then
    compHalfStep = compression
  end

  if eosvmax_tmp ~= 0 and vnewc > eosvmax_tmp then
    p_old        = 0
    compression  = 0
    compHalfStep = 0
  end

  var peqbvpb = calcEnergyForElems(p_old, e_old, q_old, compression, compHalfStep, vnewc, work, delvc, qqtmp, qltmp, eosvmax_tmp, rho0)

  var p_new = peqbvpb[0]
  var e_new = peqbvpb[1]
  var q_new = peqbvpb[2]
  var bvc   = peqbvpb[3]
  var pbvc  = peqbvpb[4]

  var ssc = calcSoundSpeedForElem(vnewc, rho0, e_new, p_new, pbvc, bvc)

  c.p  = p_new
  c.e  = e_new
  c.q  = q_new
  c.ss = ssc
end

local liszt applyMaterialPropertiesAndUpdateVolume(c : grid.cells)
  calcMonotonicQRegionForElem(c)
  evalEOSForElem(c)
  updateVolumesForElem(c)
end

local liszt updateVolumeForElements(c : grid.cells)
  var tmpV = c.vnew
  if fabs(tmpV - 1.0) < m.v_cut then tmpV = 1.0 end
  c.vol = tmpV
end

function lagrangeElements ()
  grid.cells:foreach(calcKinemAndMQGradientsForElems1)
  grid.cells:foreach(calcKinemAndMQGradientsForElems2)
  grid.cells:foreach(applyMaterialPropertiesAndUpdateVolume)
end


-- used to be called timeConstraintKernel
local liszt reduceDtCourant (c : grid.cells)
  -- courant constraint calculation
  var qqc_tmp : L.double = m.qqc
  var qqc2    = 64.0 * qqc_tmp * qqc_tmp

  var ssc       = c.ss
  var vdovtmp   = c.vdov
  var arealgtmp = c.arealg

  var dtf = ssc*ssc
  if vdovtmp < 0.0 then dtf += qqc2 * arealgtmp * arealgtmp * vdovtmp * vdovtmp end

  dtf = L.sqrt(dtf)
  dtf = arealgtmp / dtf
  if vdovtmp ~= 0.0 then
    dtcourant_tmp min= dtf
  end

  --vdovtmp = c.vdov
  --if vdovtmp ~= 0.0 then
  --  var dtdvov = m.dvovmax / (fabs(vdovtmp) + 1.e-20)
  --  dthydro_tmp min= dtdvov
  --end
end

local liszt reduceDtHydro (c : grid.cells)
  var vdovtmp = c.vdov
  if vdovtmp ~= 0.0 then
    var dtdvov = m.dvovmax / (fabs(vdovtmp) + 1.e-20)
    dthydro_tmp min= dtdvov
  end
end

function calcTimeConstraintsForElems ()
  dthydro_tmp:set(1e20)
  dtcourant_tmp:set(1e20)

  grid.cells:foreach(reduceDtCourant)
  grid.cells:foreach(reduceDtHydro)

  local courant_result = dtcourant_tmp:get()
  if courant_result ~= 1e20 then
    m.dtcourant = courant_result
  end

  local dthydro_result = dthydro_tmp:get()
  if dthydro_result ~= 1e20 then
    m.dthydro = dthydro_result
  end
end

function lagrangeLeapFrog ()
  -- calculate nodal forces, accelerations, velocities, positions, with
  -- applied boundary conditions and slide surface considerations
  lagrangeNodal()

  -- calculate element quantities (i.e. velocity gradient & q), and update
  -- material states
  lagrangeElements()

  calcTimeConstraintsForElems()
end


------------------------------------------------------------------------------------------
--[[ Solver                                                                           ]]--
------------------------------------------------------------------------------------------
local function runSolver ()
  m.initMeshParameters()
  --start_time = terralib.currenttimeinseconds()
  while m.cycle < 100 do
  --while m.time < m.stoptime do
    if m.cycle == 1 then start_time = terralib.currenttimeinseconds() end
    timeIncrement()
    lagrangeLeapFrog()
    if m.cycle % 10 == 0 then
      print("Cycle " .. tostring(m.cycle) .. ": " .. tostring(m.time) .. 's')
    end
  end
  end_time = terralib.currenttimeinseconds()
end

local function printStats()
  local average_iteration_time = (end_time - start_time) / (m.cycle-1) * 1e3
  print("[[Note that timing started AFTER first iteration]]")
  print("Total elapsed time = " .. tostring(end_time - start_time))
  print("  ms per iteration = " .. tostring(average_iteration_time))
  print("   Problem size        = " .. tostring(N))
  print("   Iteration count     = " .. tostring(m.cycle))

  -- Look up energy of origin cell
  grid.cells:MoveTo(L.CPU)
  local finalEnergy = grid.cells.e:DataPtr()[0]
  print("   Final origin energy = " .. tostring(finalEnergy))

  print()
  L.PrintStats()
end


runSolver()
printStats()
