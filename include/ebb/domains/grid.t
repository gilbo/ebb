import "ebb"

local Grid = {}
package.loaded["ebb.domains.grid"] = Grid

-------------------------------------------------------------------------------

local int_floor = L.NewMacro(function(v)
  return ebb ` L.int(L.floor(v))
end)
local max_impl = L.NewMacro(function(a,b)
  return ebb quote
    var ret = a
    if b > a then ret = b end
  in
    ret
  end
end)
local min_impl = L.NewMacro(function(a,b)
  return ebb quote
    var ret = a
    if b < a then ret = b end
  in
    ret
  end
end)

local clamp_impl = L.NewMacro(function(x, lower, upper)
  return ebb `max_impl(lower, min_impl(upper, x))
end)

-- convert a potentially continuous signed value x to
-- an address modulo the given uint m
local float_to_uint64_mod = L.NewMacro(function(x, m)
  return ebb `L.uint64(L.fmod(x,m) + m) % m
end)

-- the way we actually use these...
local wrap_idx  = float_to_uint64_mod
local clamp_idx = L.NewMacro(function(x, limit)
  return ebb `L.uint64(clamp_impl(x, 0.0, L.double(limit-1)))
end)

local function copy_table(tbl)
  local cpy = {}
  for k,v in pairs(tbl) do cpy[k] = v end
  return cpy
end

local function is_num(obj) return type(obj) == 'number' end
local function is_bool(obj) return type(obj) == 'boolean' end

-------------------------------------------------------------------------------

local Grid2d = {}
local Grid3d = {}
Grid2d.__index = Grid2d
Grid3d.__index = Grid3d

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--                                  2d Grid
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local function setup2dCells(grid)
  local Cx, Cy  = grid:xSize(), grid:ySize()
  local xw, yw  = grid:xCellWidth(), grid:yCellWidth()
  local xo, yo  = grid:xOrigin(), grid:yOrigin()
  local xn_bd   = grid:xBoundaryDepth()
  local yn_bd   = grid:yBoundaryDepth()
  local cells   = grid.cells

  -- offset
  cells:NewFieldMacro('__apply_macro', L.NewMacro(function(c,x,y)
    return ebb `L.Affine(cells, {{1,0,x},
                                 {0,1,y}}, c)                   end))

  -- Boundary/Interior subsets
  local function is_bd(x,y)
    return y <  yn_bd or y >= Cy-yn_bd or
           x <  xn_bd or x >= Cx-xn_bd
  end
  cells:NewSubsetFromFunction('boundary', is_bd)
  cells:NewSubsetFromFunction('interior', function(xi,yi)
    return not is_bd(xi,yi)
  end)

  cells:NewFieldMacro('center', L.NewMacro(function(c)
    return ebb ` L.vec2f({ xo + xw * (L.double(L.xid(c)) + 0.5),
                           yo + yw * (L.double(L.yid(c)) + 0.5) })  end))

  local xsnap = grid:xUsePeriodic() and wrap_idx or clamp_idx
  local ysnap = grid:yUsePeriodic() and wrap_idx or clamp_idx
  grid.cell_locate = L.NewMacro(function(xy_vec)
    return ebb quote
      var xy    = xy_vec -- prevent duplication
      var xval  = (xy[0] - xo)/xw
      var yval  = (xy[1] - yo)/yw
      var xidx  = xsnap(xval, Cx)
      var yidx  = ysnap(xval, Cy)
    in
      L.UNSAFE_ROW({xidx, yidx}, grid.cells)
    end
  end)

  -- boundary depths
  cells:NewFieldMacro('xneg_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(xn_bd - L.xid(c)), 0)            end))
  cells:NewFieldMacro('xpos_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(L.xid(c) - (Cx-1 - xn_bd)), 0)   end))
  cells:NewFieldMacro('yneg_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(yn_bd - L.yid(c)), 0)            end))
  cells:NewFieldMacro('ypos_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(L.yid(c) - (Cy-1 - yn_bd)), 0)   end))

  cells:NewFieldMacro('in_boundary', L.NewMacro(function(c)
    return ebb ` c.xneg_depth > 0 or c.xpos_depth > 0 or
                 c.yneg_depth > 0 or c.ypos_depth > 0           end))
  cells:NewFieldMacro('in_interior', L.NewMacro(function(c)
    return ebb ` not c.in_boundary                              end))
end

-------------------------------------------------------------------------------

local function setup2dDualCells(grid)
  local xp, yp  = grid:xUsePeriodic(), grid:yUsePeriodic()
  local Vx, Vy  = grid._vn_xy[1], grid._vn_xy[2]
  local xw, yw  = grid:xCellWidth(), grid:yCellWidth()
  local xo, yo  = grid:xOrigin(), grid:yOrigin()
  local xn_bd   = grid:xBoundaryDepth()
  local yn_bd   = grid:yBoundaryDepth()
  local dcells  = grid.dual_cells

  dcells:NewFieldMacro('__apply_macro', L.NewMacro(function(dc,x,y)
    return ebb `L.Affine(dcells, {{1,0,x},
                                  {0,1,y}}, dc)                 end))

  if not xp and not yp then
    dcells:NewFieldMacro('center', L.NewMacro(function(dc)
      return ebb `L.vec2f({ xo + xw * (L.double(L.xid(dc))),
                            yo + yw * (L.double(L.yid(dc))) })  end))
  end

  local xsnap = grid:xUsePeriodic() and wrap_idx or clamp_idx
  local ysnap = grid:yUsePeriodic() and wrap_idx or clamp_idx
  grid.dual_locate = L.NewMacro(function(xy_vec)
    return ebb quote
      var xy    = xy_vec -- prevent duplication
      var xval  = (xy[0] - xo)/xw + 0.5
      var yval  = (xy[1] - yo)/yw + 0.5
      var xidx  = xsnap(xval, Vx)
      var yidx  = ysnap(yval, Vy)
    in
      L.UNSAFE_ROW({xidx, yidx}, dcells)
    end
  end)
end

-------------------------------------------------------------------------------

local function setup2dVertices(grid)
  local Vx, Vy  = grid._vn_xy[1], grid._vn_xy[2]
  local xn_bd   = grid:xBoundaryDepth()
  local yn_bd   = grid:yBoundaryDepth()
  local verts   = grid.vertices

  verts:NewFieldMacro('__apply_macro', L.NewMacro(function(v,x,y)
    return ebb `L.Affine(verts, {{1,0,x},
                                 {0,1,y}}, v)                   end))

  -- Boundary/Interior subsets
  local function is_bd(x,y)
    return y < yn_bd or y >= Vy-yn_bd or
           x < xn_bd or x >= Vx-xn_bd
  end
  verts:NewSubsetFromFunction('boundary', is_bd)
  verts:NewSubsetFromFunction('interior', function(xi,yi)
    return not is_bd(xi,yi)
  end)

  -- boundary depths
  verts:NewFieldMacro('xneg_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(xn_bd - L.xid(v)), 0)            end))
  verts:NewFieldMacro('xpos_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(L.xid(v) - (Vx-1 - xn_bd)), 0)   end))
  verts:NewFieldMacro('yneg_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(yn_bd - L.yid(v)), 0)            end))
  verts:NewFieldMacro('ypos_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(L.yid(v) - (Vy-1 - yn_bd)), 0)   end))

  verts:NewFieldMacro('in_boundary', L.NewMacro(function(v)
    return ebb ` v.xneg_depth > 0 or v.xpos_depth > 0 or
                 v.yneg_depth > 0 or v.ypos_depth > 0           end))
  verts:NewFieldMacro('in_interior', L.NewMacro(function(v)
    return ebb ` not v.in_boundary                              end))
end
local function setup2dDualVertices(grid)
  local dverts            = grid.dual_vertices

  dverts:NewFieldMacro('__apply_macro', L.NewMacro(function(dv,x,y)
    return ebb `L.Affine(dverts, {{1,0,x},
                                  {0,1,y}}, dv)                 end))
end

-------------------------------------------------------------------------------

local function setup2dEdges(grid)
  local xp, yp  = grid:xUsePeriodic(), grid:yUsePeriodic()
  local Cx, Cy  = grid:xSize(),   grid:ySize()
  local Vx, Vy  = grid._vn_xy[1], grid._vn_xy[2]
  local verts   = grid.vertices

  grid.xedges = L.NewRelation { name = 'xedges',
                                dims = {Cx,Vy}, periodic = {xp,yp} }
  grid.yedges = L.NewRelation { name = 'yedges',
                                dims = {Vx,Cy}, periodic = {xp,yp} }
  
  -- Access vertices from the edges
  grid.xedges:NewFieldMacro('tail', L.NewMacro(function(xe)
    return ebb `L.Affine(verts, {{1,0,0},
                                 {0,1,0}}, xe)                  end))
  grid.xedges:NewFieldMacro('head', L.NewMacro(function(xe)
    return ebb `L.Affine(verts, {{1,0,1},
                                 {0,1,0}}, xe)                  end))

  grid.yedges:NewFieldMacro('tail', L.NewMacro(function(ye)
    return ebb `L.Affine(verts, {{1,0,0},
                                 {0,1,0}}, ye)                  end))
  grid.yedges:NewFieldMacro('head', L.NewMacro(function(ye)
    return ebb `L.Affine(verts, {{1,0,0},
                                 {0,1,1}}, ye)                  end))
  
  -- Access the edges from the vertices
  verts:NewFieldMacro('xpos_edge', L.NewMacro(function(v)
    return ebb `L.Affine(grid.xedges, {{1,0,0},
                                       {0,1,0}}, v)             end))
  verts:NewFieldMacro('xneg_edge', L.NewMacro(function(v)
    return ebb `L.Affine(grid.xedges, {{1,0,-1},
                                       {0,1,0}}, v)             end))

  verts:NewFieldMacro('ypos_edge', L.NewMacro(function(v)
    return ebb `L.Affine(grid.yedges, {{1,0,0},
                                       {0,1,0}}, v)             end))
  verts:NewFieldMacro('yneg_edge', L.NewMacro(function(v)
    return ebb `L.Affine(grid.yedges, {{1,0,0},
                                       {0,1,-1}}, v)            end))
end

local function setup2dFaces(grid)
  local xp, yp  = grid:xUsePeriodic(), grid:yUsePeriodic()
  local Cx, Cy  = grid:xSize(),   grid:ySize()
  local Vx, Vy  = grid._vn_xy[1], grid._vn_xy[2]
  local cells   = grid.cells

  grid.xfaces = L.NewRelation { name = 'xfaces',
                                dims = {Vx,Cy}, periodic = {xp,yp} }
  grid.yfaces = L.NewRelation { name = 'yfaces',
                                dims = {Cx,Vy}, periodic = {xp,yp} }
  
  -- Access the cells from the faces
  grid.xfaces:NewFieldMacro('pos', L.NewMacro(function(xf)
    return ebb `L.Affine(cells, {{1,0,0},
                                 {0,1,0}}, xf)                  end))
  grid.xfaces:NewFieldMacro('neg', L.NewMacro(function(xf)
    return ebb `L.Affine(cells, {{1,0,-1},
                                 {0,1,0}}, xf)                  end))

  grid.yfaces:NewFieldMacro('pos', L.NewMacro(function(yf)
    return ebb `L.Affine(cells, {{1,0,0},
                                 {0,1,0}}, yf)                  end))
  grid.yfaces:NewFieldMacro('neg', L.NewMacro(function(yf)
    return ebb `L.Affine(cells, {{1,0,0},
                                 {0,1,-1}}, yf)                 end))
  
  -- Access the faces from the cells
  cells:NewFieldMacro('xpos_face', L.NewMacro(function(v)
    return ebb `L.Affine(grid.xfaces, {{1,0,1},
                                       {0,1,0}}, v)             end))
  cells:NewFieldMacro('xneg_face', L.NewMacro(function(v)
    return ebb `L.Affine(grid.xfaces, {{1,0,0},
                                       {0,1,0}}, v)             end))

  cells:NewFieldMacro('ypos_face', L.NewMacro(function(v)
    return ebb `L.Affine(grid.yfaces, {{1,0,0},
                                       {0,1,1}}, v)             end))
  cells:NewFieldMacro('yneg_face', L.NewMacro(function(v)
    return ebb `L.Affine(grid.yfaces, {{1,0,0},
                                       {0,1,0}}, v)             end))

  -- Access the corresponding edge from the face and vice-versa
  grid.xfaces:NewFieldMacro('edge', L.NewMacro(function(xf)
    return ebb `L.Affine(grid.yedges, {{1,0,0},
                                       {0,1,0}}, xf)            end))
  grid.yfaces:NewFieldMacro('edge', L.NewMacro(function(yf)
    return ebb `L.Affine(grid.xedges, {{1,0,0},
                                       {0,1,0}}, yf)            end))

  grid.xedges:NewFieldMacro('face', L.NewMacro(function(xe)
    return ebb `L.Affine(grid.yfaces, {{1,0,0},
                                       {0,1,0}}, xe)            end))
  grid.yedges:NewFieldMacro('face', L.NewMacro(function(ye)
    return ebb `L.Affine(grid.xfaces, {{1,0,0},
                                       {0,1,0}}, ye)            end))
end

-------------------------------------------------------------------------------

local function setup2dInterconnects(grid)
  local Cx, Cy  = grid:xSize(),   grid:ySize()
  local Vx, Vy  = grid._vn_xy[1], grid._vn_xy[2]
  local cells   = grid.cells
  local verts   = grid.vertices
  local dcells  = grid.dual_cells
  local dverts  = grid.dual_vertices

  -- v <-> dc
  dcells:NewFieldMacro('vertex', L.NewMacro(function(dc)
    return ebb `L.Affine(verts, {{1,0,0},
                                 {0,1,0}}, dc)                  end))
  verts:NewFieldMacro('dual_cell', L.NewMacro(function(v)
    return ebb `L.Affine(dcells, {{1,0,0},
                                  {0,1,0}}, v)                  end))

  -- v <-> c
  cells:NewFieldMacro('vertex', L.NewMacro(function(c)
    return ebb `L.Affine(verts, {{1,0,0},
                                 {0,1,0}}, c)                   end))
  verts:NewFieldMacro('cell', L.NewMacro(function(v)
    return ebb `L.Affine(cells, {{1,0,0},
                                 {0,1,0}}, v)                   end))

  -- dv <-> c
  cells:NewFieldMacro('dual_vertex', L.NewMacro(function(c)
      return ebb `L.Affine(dverts, {{1,0,0},
                                    {0,1,0}}, c)                end))
  dverts:NewFieldMacro('cell', L.NewMacro(function(dv)
      return ebb `L.Affine(cells, {{1,0,0},
                                   {0,1,0}}, dv)                end))

  -- dv <-> dc
  dcells:NewFieldMacro('dual_vertex', L.NewMacro(function(dc)
      return ebb `L.Affine(dverts, {{1,0,0},
                                    {0,1,0}}, dc)               end))
  dverts:NewFieldMacro('dual_cell', L.NewMacro(function(dv)
      return ebb `L.Affine(dcells, {{1,0,0},
                                    {0,1,0}}, dv)               end))
end

-------------------------------------------------------------------------------

function Grid.NewGrid2d(params)
  local calling_convention = [[

NewGrid2d should be called with named parameters:
Grid.NewGrid2d{
  size          = {#,#},            -- number of cells in x and y
  origin        = {#,#},            -- x,y coordinates of grid origin
  width         = {#,#},            -- x,y width of grid in coordinate system
  (optional)
  boundary_depth    = {#,#},        -- depth of boundary region (default value: {1,1})
  periodic_boundary = {bool,bool},  -- use periodic boundary conditions (default value: {false,false})
}]]
  local function check_params(params)
    if type(params) ~= 'table' then return false end
    if type(params.size) ~= 'table' or
       type(params.origin) ~= 'table' or
       type(params.width) ~= 'table' then return false end
    local check = is_num(params.size[1])    and is_num(params.size[2]) and
                  is_num(params.origin[1])  and is_num(params.origin[2]) and
                  is_num(params.width[1])   and is_num(params.width[2])
    local bd = params.boundary_depth
    local pb = params.periodic_boundary
    if bd then check = check and type(bd) == 'table' and
                                 is_num(bd[1]) and is_num(bd[2]) end
    if pb then check = check and type(pb) == 'table' and
                                 is_bool(pb[1]) and is_bool(pb[2]) end
    return check
  end
  if not check_params(params) then error(calling_convention, 2) end

  -- default parameters
  local pb      = params.periodic_boundary or {false, false}
  local bd      = params.boundary_depth    or {1, 1}
  local vsize   = copy_table(params.size)
  for i=1,2 do
    if pb[i] then bd[i] = 0
             else vsize[i] = vsize[i] + 1 end
  end

  local grid = setmetatable({
    _vn_xy      = vsize, -- already copied
    _n_xy       = copy_table(params.size),
    _origin     = copy_table(params.origin),
    _dims       = copy_table(params.width),
    _bd_depth   = bd,
    _periodic   = pb,
    -- relations
    cells           = L.NewRelation { name = 'cells', 
                                      dims = params.size,  periodic = pb },
    dual_vertices   = L.NewRelation { name = 'dual_vertices',
                                      dims = params.size,  periodic = pb },
    vertices        = L.NewRelation { name = 'vertices',
                                      dims = vsize,        periodic = pb },
    dual_cells      = L.NewRelation { name = 'dual_cells',
                                      dims = vsize,        periodic = pb },
    --faces           = L.NewRelation { name = 'faces',
    --                                  dims = vsize,        periodic = pb },
  }, Grid2d)

  setup2dCells(grid)
  setup2dDualCells(grid)
  setup2dVertices(grid)
  setup2dDualVertices(grid)
  setup2dInterconnects(grid)
  setup2dEdges(grid)
  setup2dFaces(grid)

  return grid
end

function Grid2d:xSize()             return self._n_xy[1]            end
function Grid2d:ySize()             return self._n_xy[2]            end
function Grid2d:xOrigin()           return self._origin[1]          end
function Grid2d:yOrigin()           return self._origin[2]          end
function Grid2d:xWidth()            return self._dims[1]            end
function Grid2d:yWidth()            return self._dims[2]            end
function Grid2d:xBoundaryDepth()    return self._bd_depth[1]        end
function Grid2d:yBoundaryDepth()    return self._bd_depth[2]        end
function Grid2d:xUsePeriodic()      return self._periodic[1]        end
function Grid2d:yUsePeriodic()      return self._periodic[2]        end
function Grid2d:xCellWidth()  return self:xWidth() / (1.0 * self:xSize()) end
function Grid2d:yCellWidth()  return self:yWidth() / (1.0 * self:ySize()) end

function Grid2d:Size()            return copy_table(self._n_xy)     end
function Grid2d:Origin()          return copy_table(self._origin)   end
function Grid2d:Width()           return copy_table(self._dims)     end
function Grid2d:BoundaryDepth()   return copy_table(self._bd_depth) end
function Grid2d:UsePeriodic()     return copy_table(self._periodic) end
function Grid2d:CellWidth()
  return { self:xCellWidth(), self:yCellWidth() }
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--                                  3d Grid
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


local function setup3dCells(grid)
  local Cx, Cy, Cz  = grid:xSize(), grid:ySize(), grid:zSize()
  local xw, yw, zw  = grid:xCellWidth(), grid:yCellWidth(), grid:zCellWidth()
  local xo, yo, zo  = grid:xOrigin(), grid:yOrigin(), grid:zOrigin()
  local xn_bd       = grid:xBoundaryDepth()
  local yn_bd       = grid:yBoundaryDepth()
  local zn_bd       = grid:zBoundaryDepth()
  local cells       = grid.cells

  -- relative offset
  cells:NewFieldMacro('__apply_macro', L.NewMacro(function(c,x,y,z)
      return ebb `L.Affine(cells, {{1,0,0,x},
                                   {0,1,0,y},
                                   {0,0,1,z}}, c)               end))

  -- Boundary/Interior subsets
  local function is_bd(xi, yi, zi)
    return  xi < xn_bd or xi >= Cx-xn_bd or
            yi < yn_bd or yi >= Cy-yn_bd or
            zi < zn_bd or zi >= Cz-zn_bd
  end
  cells:NewSubsetFromFunction('boundary', is_bd)
  cells:NewSubsetFromFunction('interior', function(xi,yi,zi)
    return not is_bd(xi,yi,zi)
  end)

  cells:NewFieldMacro('center', L.NewMacro(function(c)
    return ebb ` L.vec3f({ xo + xw * (L.double(L.xid(c)) + 0.5),
                           yo + yw * (L.double(L.yid(c)) + 0.5),
                           zo + zw * (L.double(L.zid(c)) + 0.5) })  end))

  local xsnap = grid:xUsePeriodic() and wrap_idx or clamp_idx
  local ysnap = grid:yUsePeriodic() and wrap_idx or clamp_idx
  local zsnap = grid:zUsePeriodic() and wrap_idx or clamp_idx
  grid.cell_locate = L.NewMacro(function(xyz_vec)
    return ebb quote
      var xyz  = xyz_vec -- prevent duplication
      var xval = (xyz[0] - xo) / xw
      var yval = (xyz[1] - yo) / yw
      var zval = (xyz[2] - zo) / zw
      var xidx = xsnap(xval, Cx)
      var yidx = ysnap(yval, Cy)
      var zidx = zsnap(zval, Cz)
    in
      L.UNSAFE_ROW({xidx, yidx, zidx}, cells)
    end
  end)

  -- boundary depths
  cells:NewFieldMacro('xneg_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(xn_bd - L.xid(c)), 0)            end))
  cells:NewFieldMacro('xpos_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(L.xid(c) - (Cx-1 - xn_bd)), 0)   end))
  cells:NewFieldMacro('yneg_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(yn_bd - L.yid(c)), 0)            end))
  cells:NewFieldMacro('ypos_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(L.yid(c) - (Cy-1 - yn_bd)), 0)   end))
  cells:NewFieldMacro('zneg_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(zn_bd - L.zid(c)), 0)            end))
  cells:NewFieldMacro('zpos_depth', L.NewMacro(function(c)
    return ebb `max_impl(L.int(L.zid(c) - (Cz-1 - zn_bd)), 0)   end))

  cells:NewFieldMacro('in_boundary', L.NewMacro(function(c)
    return ebb ` c.xneg_depth > 0 or c.xpos_depth > 0 or
                 c.yneg_depth > 0 or c.ypos_depth > 0 or
                 c.zneg_depth > 0 or c.zpos_depth > 0
  end))
  cells:NewFieldMacro('in_interior', L.NewMacro(function(c)
    return ebb ` not c.in_boundary
  end))
end

-------------------------------------------------------------------------------

local function setup3dDualCells(grid)
  local xp, yp, zp  = grid:xUsePeriodic(), grid:yUsePeriodic(),
                      grid:zUsePeriodic()
  local Vx, Vy, Vz  = grid._vn_xyz[1], grid._vn_xyz[2], grid._vn_xyz[3]
  local xw, yw, zw  = grid:xCellWidth(), grid:yCellWidth(), grid:zCellWidth()
  local xo, yo, zo  = grid:xOrigin(), grid:yOrigin(), grid:zOrigin()
  local xn_bd       = grid:xBoundaryDepth()
  local yn_bd       = grid:yBoundaryDepth()
  local zn_bd       = grid:zBoundaryDepth()
  local dcells      = grid.dual_cells

  -- relative offset
  dcells:NewFieldMacro('__apply_macro', L.NewMacro(function(dc,x,y,z)
      return ebb `L.Affine(dcells, {{1,0,0,x},
                                    {0,1,0,y},
                                    {0,0,1,z}}, dc)             end))

  if not xp and not yp and not zp then
    dcells:NewFieldMacro('center', L.NewMacro(function(dc)
      return ebb `L.vec3f({ xo + xw * (L.double(L.xid(dc))),
                            yo + yw * (L.double(L.yid(dc))),
                            zo + zw * (L.double(L.zid(dc))) })  end))
  end

  local xsnap = grid:xUsePeriodic() and wrap_idx or clamp_idx
  local ysnap = grid:yUsePeriodic() and wrap_idx or clamp_idx
  local zsnap = grid:zUsePeriodic() and wrap_idx or clamp_idx
  grid.dual_locate = L.NewMacro(function(xyz_vec)
    return ebb quote
      var xyz = xyz_vec -- prevent duplication
      var xval = (xyz[0] - xo) / xw + 0.5
      var yval = (xyz[1] - yo) / yw + 0.5
      var zval = (xyz[2] - zo) / zw + 0.5
      var xidx = xsnap(xval, Vx)
      var yidx = ysnap(yval, Vy)
      var zidx = zsnap(zval, Vz)
    in
      L.UNSAFE_ROW({xidx, yidx, zidx}, dcells)
    end
  end)
end

-------------------------------------------------------------------------------

local function setup3dVertices(grid)
  local xp, yp, zp  = grid:xUsePeriodic(), grid:yUsePeriodic(),
                      grid:zUsePeriodic()
  local Vx, Vy, Vz  = grid._vn_xyz[1], grid._vn_xyz[2], grid._vn_xyz[3]
  local xw, yw, zw  = grid:xCellWidth(), grid:yCellWidth(), grid:zCellWidth()
  local xo, yo, zo  = grid:xOrigin(), grid:yOrigin(), grid:zOrigin()
  local xn_bd       = grid:xBoundaryDepth()
  local yn_bd       = grid:yBoundaryDepth()
  local zn_bd       = grid:zBoundaryDepth()
  local verts       = grid.vertices

  -- relative offset
  verts:NewFieldMacro('__apply_macro', L.NewMacro(function(v,x,y,z)
      return ebb `L.Affine(verts, {{1,0,0,x},
                                   {0,1,0,y},
                                   {0,0,1,z}}, v)               end))

  -- Boundary/Interior subsets
  local function is_bd(xi,yi,zi)
    return  xi < xn_bd or xi >= Vx-xn_bd or
            yi < yn_bd or yi >= Vy-yn_bd or
            zi < zn_bd or zi >= Vz-zn_bd
  end
  verts:NewSubsetFromFunction('boundary', is_bd)
  verts:NewSubsetFromFunction('interior', function(xi,yi,zi)
    return not is_bd(xi,yi,zi)
  end)

  -- boundary depths
  verts:NewFieldMacro('xneg_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(xn_bd - L.xid(v)), 0)            end))
  verts:NewFieldMacro('xpos_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(L.xid(v) - (Vx-1 - xn_bd)), 0)   end))
  verts:NewFieldMacro('yneg_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(yn_bd - L.yid(v)), 0)            end))
  verts:NewFieldMacro('ypos_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(L.yid(v) - (Vy-1 - yn_bd)), 0)   end))
  verts:NewFieldMacro('zneg_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(zn_bd - L.zid(v)), 0)            end))
  verts:NewFieldMacro('zpos_depth', L.NewMacro(function(v)
    return ebb `max_impl(L.int(L.zid(v) - (Vz-1 - zn_bd)), 0)   end))

  verts:NewFieldMacro('in_boundary', L.NewMacro(function(v)
    return ebb ` v.xneg_depth > 0 or v.xpos_depth > 0 or
                 v.yneg_depth > 0 or v.ypos_depth > 0 or
                 v.zneg_depth > 0 or v.zpos_depth > 0
  end))
  verts:NewFieldMacro('in_interior', L.NewMacro(function(v)
    return ebb ` not v.in_boundary
  end))
end

local function setup3dDualVertices(grid)
  grid.dual_vertices:NewFieldMacro('__apply_macro',
    L.NewMacro(function(dv,xoff,yoff,zoff)
      return ebb `L.Affine(grid.dual_vertices, {{1,0,0,xoff},
                                                {0,1,0,yoff},
                                                {0,0,1,zoff}}, dv)
    end))
end

-------------------------------------------------------------------------------

-------------------------------------------------------------------------------

local function setup3dInterconnects(grid)
  local Cx, Cy, Cz  = grid:xSize(), grid:ySize(), grid:zSize()
  local Vx, Vy, Vz  = grid._vn_xyz[1], grid._vn_xyz[2], grid._vn_xyz[3]
  local cells       = grid.cells
  local verts       = grid.vertices
  local dcells      = grid.dual_cells
  local dverts      = grid.dual_vertices

  -- v <-> dc
  dcells:NewFieldMacro('vertex', L.NewMacro(function(dc)
      return ebb `L.Affine(verts, {{1,0,0,0},
                                   {0,1,0,0},
                                   {0,0,1,0}}, dc)              end))
  verts:NewFieldMacro('dual_cell', L.NewMacro(function(v)
      return ebb `L.Affine(dcells, {{1,0,0,0},
                                    {0,1,0,0},
                                    {0,0,1,0}}, v)              end))

  -- v <-> c
  cells:NewFieldMacro('vertex', L.NewMacro(function(c)
      return ebb `L.Affine(verts, {{1,0,0,0},
                                   {0,1,0,0},
                                   {0,0,1,0}}, c)               end))
  verts:NewFieldMacro('cell', L.NewMacro(function(v)
      return ebb `L.Affine(cells, {{1,0,0,0},
                                   {0,1,0,0},
                                   {0,0,1,0}}, v)               end))

  -- dv <-> c
  cells:NewFieldMacro('dual_vertex', L.NewMacro(function(c)
      return ebb `L.Affine(dverts, {{1,0,0,0},
                                    {0,1,0,0},
                                    {0,0,1,0}}, c)              end))
  dverts:NewFieldMacro('cell', L.NewMacro(function(dv)
      return ebb `L.Affine(cells, {{1,0,0,0},
                                   {0,1,0,0},
                                   {0,0,1,0}}, dv)              end))

  -- dv <-> dc
  dcells:NewFieldMacro('dual_vertex', L.NewMacro(function(dc)
      return ebb `L.Affine(dverts, {{1,0,0,0},
                                    {0,1,0,0},
                                    {0,0,1,0}}, dc)             end))
  dverts:NewFieldMacro('dual_cell', L.NewMacro(function(dv)
      return ebb `L.Affine(dcells, {{1,0,0,0},
                                    {0,1,0,0},
                                    {0,0,1,0}}, dv)             end))
end

-------------------------------------------------------------------------------

function Grid.NewGrid3d(params)
  local calling_convention = [[

NewGrid3d should be called with named parameters:
Grid.NewGrid3d{
  size          = {#,#,#},          -- number of cells in x,y, and z
  origin        = {#,#,#},          -- x,y,z coordinates of grid origin
  width         = {#,#,#},          -- x,y,z width of grid in coordinate system
    (optional)
  boundary_depth    = {#,#,#},      -- depth of boundary region
                                        (default value: {1,1,1})
  periodic_boundary = {bool,bool,bool},
                                    -- use periodic boundary conditions
                                        (default value: {false,false,false})
}]]
  local function check_params(params)
    if type(params) ~= 'table' then return false end
    if type(params.size) ~= 'table' or
       type(params.origin) ~= 'table' or
       type(params.width) ~= 'table' then return false end
    local check = true
    for i=1,3 do
      check = check and is_num(params.size[i])
                    and is_num(params.origin[i])
                    and is_num(params.width[i])
    end
    local bd = params.boundary_depth
    local pb = params.periodic_boundary
    if bd then check = check and type(bd) == 'table' and is_num(bd[1]) and
                                       is_num(bd[2]) and is_num(bd[3]) end
    if pb then check = check and type(pb) == 'table' and is_bool(pb[1]) and
                                      is_bool(pb[2]) and is_bool(pb[3]) end
    return check
  end
  if not check_params(params) then error(calling_convention, 2) end

  -- default
  local wrap_bnd      = params.periodic_boundary or {false, false, false}
  local bnd_depth     = params.boundary_depth    or {1, 1, 1}
  local vsize         = copy_table(params.size)
  for i=1,3 do
    if wrap_bnd[i] then bnd_depth[i] = 0
                   else vsize[i] = vsize[i] + 1 end
  end

  local grid = setmetatable({
    _vn_xyz     = vsize, -- just for internal use, not exposed
    _n_xyz      = copy_table(params.size),
    _origin     = copy_table(params.origin),
    _dims       = copy_table(params.width),
    _bd_depth   = bnd_depth,
    _periodic   = wrap_bnd,
    -- relations
    cells           = L.NewRelation { name = 'cells',
                        dims = params.size,  periodic = wrap_bnd },
    dual_vertices   = L.NewRelation { name = 'dual_vertices',
                        dims = params.size,  periodic = wrap_bnd },
    vertices        = L.NewRelation { name = 'vertices',
                        dims = vsize,        periodic = wrap_bnd },
    dual_cells      = L.NewRelation { name = 'dual_cells',
                        dims = vsize,        periodic = wrap_bnd },
  }, Grid3d)

  setup3dCells(grid)
  setup3dDualCells(grid)
  setup3dVertices(grid)
  setup3dDualVertices(grid)
  setup3dInterconnects(grid)

  return grid
end



function Grid3d:xSize()             return self._n_xyz[1]           end
function Grid3d:ySize()             return self._n_xyz[2]           end
function Grid3d:zSize()             return self._n_xyz[3]           end
function Grid3d:xOrigin()           return self._origin[1]          end
function Grid3d:yOrigin()           return self._origin[2]          end
function Grid3d:zOrigin()           return self._origin[3]          end
function Grid3d:xWidth()            return self._dims[1]            end
function Grid3d:yWidth()            return self._dims[2]            end
function Grid3d:zWidth()            return self._dims[3]            end
function Grid3d:xBoundaryDepth()    return self._bd_depth[1]        end
function Grid3d:yBoundaryDepth()    return self._bd_depth[2]        end
function Grid3d:zBoundaryDepth()    return self._bd_depth[3]        end
function Grid3d:xUsePeriodic()      return self._periodic[1]        end
function Grid3d:yUsePeriodic()      return self._periodic[2]        end
function Grid3d:zUsePeriodic()      return self._periodic[3]        end
function Grid3d:xCellWidth()  return self:xWidth() / (1.0 * self:xSize()) end
function Grid3d:yCellWidth()  return self:yWidth() / (1.0 * self:ySize()) end
function Grid3d:zCellWidth()  return self:zWidth() / (1.0 * self:zSize()) end

function Grid3d:Size()            return copy_table(self._n_xyz)    end
function Grid3d:Origin()          return copy_table(self._origin)   end
function Grid3d:Width()           return copy_table(self._dims)     end
function Grid3d:BoundaryDepth()   return copy_table(self._bd_depth) end
function Grid3d:UsePeriodic()     return copy_table(self._periodic) end
function Grid3d:CellWidth()
  return { self:xCellWidth(), self:yCellWidth(), self:zCellWidth() }
end




