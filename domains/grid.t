import "compiler.liszt"

local Grid = {}
package.loaded["domains.grid"] = Grid
local cmath = terralib.includecstring '#include "math.h"'

local L = terralib.require "compiler.lisztlib"

local Grid2d = {}
local Grid3d = {}
Grid2d.__index = Grid2d
Grid3d.__index = Grid3d

local int_floor = L.NewMacro(function(v)
    return liszt ` L.int(cmath.floor(v))
end)
local max_impl = L.NewMacro(function(a,b)
    return liszt quote
        var ret = a
        if b > a then ret = b end
    in
        ret
    end
end)
local min_impl = L.NewMacro(function(a,b)
    return liszt quote
        var ret = a
        if b < a then ret = b end
    in
        ret
    end
end)

local clamp_impl = L.NewMacro(function(x, lower, upper)
    return liszt `max_impl(lower, min_impl(upper, x))
end)

-- convert a potentially continuous signed value x to
-- an address modulo the given uint m
local float_to_addr_mod = L.NewMacro(function(x, m)
    return liszt quote
        var value = x
        var result : L.addr
        if value < 0 then
            result = int_floor(value) % m + m
        else
            result = int_floor(value) % m
        end
    in
        result
    end
end)


-- There are N x M cells for an NxM grid
local function setup2dCells(grid)
    local xsize, ysize      = grid:xSize(), grid:ySize()
    local cell_width        = grid:cellWidth()
    local cell_height       = grid:cellHeight()
    local xorigin           = grid:xOrigin()
    local yorigin           = grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

    -- relative offset
    grid.cells:NewFieldMacro('__apply_macro',
        L.NewMacro(function(c,xoff,yoff)
            return liszt quote
                var xp = (xoff + xsize)
                var yp = (yoff + ysize)
                var xi = L.addr((xp + c.xid) % xsize)
                var yi = L.addr((yp + c.yid) % ysize)
            in
                L.UNSAFE_ROW( xi + yi * xsize, grid.cells )
            end
        end))

    -- Boundary/Interior subsets
    local function is_boundary(i)
        return
            math.floor(i/xsize) <  yn_bd or
            math.floor(i/xsize) >= ysize-yn_bd or
                        i%xsize <  xn_bd or
                        i%xsize >= xsize-xn_bd
    end
    grid.cells:NewSubsetFromFunction('boundary', is_boundary)
    grid.cells:NewSubsetFromFunction('interior', function(i)
        return not is_boundary(i)
    end)
    
    -- Should these be hidden?
    grid.cells:NewFieldMacro('xid', L.NewMacro(function(c)
        return liszt ` L.id(c) % L.addr(xsize)
    end))
    grid.cells:NewFieldMacro('yid', L.NewMacro(function(c)
        return liszt ` L.id(c) / L.addr(xsize)
    end))

    grid.cells:NewFieldMacro('center', L.NewMacro(function(c)
        return liszt ` L.vec2f({
            xorigin + cell_width  * (L.double(c.xid) + 0.5),
            yorigin + cell_height * (L.double(c.yid) + 0.5) })
    end))

    grid.cell_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xy = xy_vec -- prevent duplication
            var xval = (xy[0] - xorigin)/cell_width
            var yval = (xy[1] - yorigin)/cell_height
            var xidx = L.addr(clamp_impl(L.int(xval), 0, xsize-1))
            var yidx = L.addr(clamp_impl(L.int(yval), 0, ysize-1))
        in
            L.UNSAFE_ROW(xidx + yidx * xsize, grid.cells)
        end
    end)

    -- boundary depths
    grid.cells:NewFieldMacro('xneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(xn_bd - c.xid), 0)
    end))
    grid.cells:NewFieldMacro('xpos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(c.xid - (xsize-1 - xn_bd)), 0)
    end))
    grid.cells:NewFieldMacro('yneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(yn_bd - c.yid), 0)
    end))
    grid.cells:NewFieldMacro('ypos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(c.yid - (ysize-1 - yn_bd)), 0)
    end))

    grid.cells:NewFieldMacro('in_boundary', L.NewMacro(function(c)
        return liszt ` c.xneg_depth > 0 or c.xpos_depth > 0 or
                       c.yneg_depth > 0 or c.ypos_depth > 0
    end))
    grid.cells:NewFieldMacro('in_interior', L.NewMacro(function(c)
        return liszt ` not c.in_boundary
    end))
end

-- There are (N+1) x (M+1) dual cells for an NxM grid
-- unless perdiodicity...
local function setup2dDualCells(grid)
    local xpd, ypd      = grid:xUsePeriodic(), grid:yUsePeriodic()
    local xsize         = grid:xSize() + (xpd and 0 or 1)
    local ysize         = grid:ySize() + (ypd and 0 or 1)
    local cell_width    = grid:cellWidth()
    local cell_height   = grid:cellHeight()
    local xorigin       = grid:xOrigin()
    local yorigin       = grid:yOrigin()
    local xn_bd         = grid:xBoundaryDepth()
    local yn_bd         = grid:yBoundaryDepth()

    -- relative offset
    grid.dual_cells:NewFieldMacro('__apply_macro',
        L.NewMacro(function(dc,xoff,yoff)
            return liszt quote
                var xp = (xoff + xsize)
                var yp = (yoff + ysize)
                var xi = L.addr((xp + c.xid) % xsize)
                var yi = L.addr((yp + c.yid) % ysize)
            in
                L.UNSAFE_ROW( xi + yi * xsize, grid.dual_cells )
            end
        end))

    -- Should these be hidden?
    grid.dual_cells:NewFieldMacro('xid', L.NewMacro(function(dc)
        return liszt ` L.id(dc) % L.addr(xsize)
    end))
    grid.dual_cells:NewFieldMacro('yid', L.NewMacro(function(dc)
        return liszt ` L.id(dc) / L.addr(xsize)
    end))
    if not xpd and not ypd then
        grid.dual_cells:NewFieldMacro('center', L.NewMacro(function(dc)
            return liszt `L.vec2f({
                xorigin +  cell_width * (L.double(dc.xid)),
                yorigin + cell_height * (L.double(dc.yid))
            })
        end))
    end

    grid.dual_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xy = xy_vec -- prevent duplication
            var xval = (xy[0] - xorigin)/cell_width + 0.5
            var yval = (xy[1] - yorigin)/cell_height + 0.5
            var xidx : L.addr
            var yidx : L.addr
            if xpd then
                xidx = float_to_addr_mod(xval, xsize)
            else
                xidx = clamp_impl(L.int(xval), 0, xsize-1)
            end
            if ypd then
                yidx = float_to_addr_mod(yval, ysize)
            else
                yidx = clamp_impl(L.int(yval), 0, ysize-1)
            end
        in
            L.UNSAFE_ROW(xidx + yidx * xsize, grid.dual_cells)
        end
    end)
end

-- There are (N+1) x (M+1) vertices for an NxM grid
-- unless perdiodicity...
local function setup2dVertices(grid)
    local xpd, ypd      = grid:xUsePeriodic(), grid:yUsePeriodic()
    local xsize         = grid:xSize() + (xpd and 0 or 1)
    local ysize         = grid:ySize() + (ypd and 0 or 1)
    local cell_width    = grid:cellWidth()
    local cell_height   = grid:cellHeight()
    local xorigin       = grid:xOrigin()
    local yorigin       = grid:yOrigin()
    local xn_bd         = grid:xBoundaryDepth()
    local yn_bd         = grid:yBoundaryDepth()

    -- relative offset
    grid.vertices:NewFieldMacro('__apply_macro',
        L.NewMacro(function(v,xoff,yoff)
            return liszt quote
                var xp = (xoff + xsize)
                var yp = (yoff + ysize)
                var xi = L.addr((xp + c.xid) % xsize)
                var yi = L.addr((yp + c.yid) % ysize)
            in
                L.UNSAFE_ROW( xi + yi * xsize, grid.vertices )
            end
        end))

    -- Should these be hidden?
    grid.vertices:NewFieldMacro('xid', L.NewMacro(function(v)
        return liszt ` L.id(v) % L.addr(xsize)
    end))
    grid.vertices:NewFieldMacro('yid', L.NewMacro(function(v)
        return liszt ` L.id(v) / L.addr(xsize)
    end))
end

local function setup2dInterconnects(grid)
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local cxsize, cysize    = grid:xSize(), grid:ySize()
    local vxsize            = cxsize + (xpd and 0 or 1)
    local vysize            = cysize + (ypd and 0 or 1)

    grid.dual_cells:NewFieldMacro('vertex', L.NewMacro(function(dc)
        return liszt ` L.UNSAFE_ROW( L.id(dc), grid.vertices )
    end))
    grid.vertices:NewFieldMacro('dual_cell', L.NewMacro(function(v)
        return liszt ` L.UNSAFE_ROW( L.id(v), grid.dual_cells )
    end))

    grid.cells:NewFieldMacro('vertex', L.NewMacro(function(c)
        return liszt `
            L.UNSAFE_ROW( c.xid + c.yid * vxsize, grid.vertices )
    end))
    grid.vertices:NewFieldMacro('cell', L.NewMacro(function(v)
        return liszt `
            L.UNSAFE_ROW( v.xid + v.yid * cxsize, grid.cells )
    end))
end

function Grid.NewGrid2d(params)
    local calling_convention = [[

NewGrid2d should be called with named parameters:
Grid.NewGrid2d{
  size          = {#,#},            -- number of cells in x and y
  origin        = {#,#},            -- x,y coordinates of grid origin
  width         = #,                -- width of grid coordinate system
  height        = #,                -- height of grid coordinate system
  (optional)
  boundary_depth    = {#,#},        -- depth of boundary region (default value: {1,1})
  periodic_boundary = {bool,bool},  -- use periodic boundary conditions (default value: {false,false})
}]]
    local function is_num(obj) return type(obj) == 'number' end
    local function is_bool(obj) return type(obj) == 'boolean' end
    local function check_params(params)
        local check =
            type(params) == 'table' and
            type(params.size) == 'table' and
            type(params.origin) == 'table' and
            is_num(params.size[1]) and is_num(params.size[2]) and
            is_num(params.origin[1]) and is_num(params.origin[2]) and
            is_num(params.width) and is_num(params.height)
        if check and params.boundary_depth then
            check = check and
                    type(params.boundary_depth) == 'table' and
                    is_num(params.boundary_depth[1]) and
                    is_num(params.boundary_depth[2])
        end
        if check and params.periodic_boundary then
            check = check and
                    type(params.periodic_boundary) == 'table' and
                    is_bool(params.periodic_boundary[1]) and
                    is_bool(params.periodic_boundary[2])
        end
        return check
    end
    if not check_params(params) then error(calling_convention, 2) end

    -- default
    params.periodic_boundary = params.periodic_boundary or {false, false}
    params.boundary_depth    = params.boundary_depth or {1, 1}
    for i=1,2 do
        if params.periodic_boundary[i] then
            params.boundary_depth[i] = 0
        end
    end

    local nCells        = params.size[1] * params.size[2]
    local dcsize        = {}
    local nDualCells    = 1
    for i=1,2 do
        if params.periodic_boundary[i] then
            dcsize[i] = params.size[i]
        else
            dcsize[i] = params.size[i] + 1
        end
        nDualCells = nDualCells * dcsize[i]
    end
    local nVerts        = nDualCells

    local grid = setmetatable({
        _n_xy       = params.size,
        _origin     = params.origin,
        _dims       = {params.width, params.height},
        _bd_depth   = params.boundary_depth,
        _periodic   = params.periodic_boundary,
        -- relations
        cells       = L.NewRelation(nCells, 'cells'),
        dual_cells  = L.NewRelation(nDualCells, 'dual_cells'),
        vertices    = L.NewRelation(nVerts, 'vertices'),
    }, Grid2d)

    setup2dCells(grid)
    setup2dDualCells(grid)
    setup2dVertices(grid)
    setup2dInterconnects(grid)

    return grid
end

function Grid2d:xSize()             return self._n_xy[1]            end
function Grid2d:ySize()             return self._n_xy[2]            end
function Grid2d:xOrigin()           return self._origin[1]          end
function Grid2d:yOrigin()           return self._origin[2]          end
function Grid2d:width()             return self._dims[1]            end
function Grid2d:height()            return self._dims[2]            end
function Grid2d:xBoundaryDepth()    return self._bd_depth[1]        end
function Grid2d:yBoundaryDepth()    return self._bd_depth[2]        end
function Grid2d:xUsePeriodic()      return self._periodic[1]        end
function Grid2d:yUsePeriodic()      return self._periodic[2]        end
function Grid2d:cellWidth()   return self:width() / (1.0 * self:xSize())  end
function Grid2d:cellHeight()  return self:height() / (1.0 * self:ySize()) end


-- There are N x M x L cells for an NxMxL grid
local function setup3dCells(grid)
    local xsize, ysize, zsize   = grid:xSize(), grid:ySize(), grid:zSize()
    local xysize                = xsize * ysize
    local xcwidth               = grid:xCellWidth()
    local ycwidth               = grid:yCellWidth()
    local zcwidth               = grid:zCellWidth()
    local xorigin               = grid:xOrigin()
    local yorigin               = grid:yOrigin()
    local zorigin               = grid:zOrigin()
    local xn_bd                 = grid:xBoundaryDepth()
    local yn_bd                 = grid:yBoundaryDepth()
    local zn_bd                 = grid:zBoundaryDepth()

    -- relative offset
    grid.cells:NewFieldMacro('__apply_macro',
        L.NewMacro(function(c,xoff,yoff,zoff)
            return liszt quote
                var xp = (xoff + xsize)
                var yp = (yoff + ysize)
                var zp = (zoff + zsize)
                var xi = L.addr((xp + c.xid) % xsize)
                var yi = L.addr((yp + c.yid) % ysize)
                var zi = L.addr((zp + c.zid) % zsize)
            in
                L.UNSAFE_ROW( xi + yi * xsize + zi * xysize, grid.cells )
            end
        end))

    -- Boundary/Interior subsets
    local function is_boundary(i)
        local xi = i%xsize
        local xq = (i-xi)/xsize
        local yi = xq%ysize
        local yq = (xq-yi)/ysize
        local zi = yq--%zsize
        return  xi < xn_bd or xi >= xsize-xn_bd or
                yi < yn_bd or yi >= ysize-yn_bd or
                zi < zn_bd or zi >= zsize-zn_bd
    end
    grid.cells:NewSubsetFromFunction('boundary', is_boundary)
    grid.cells:NewSubsetFromFunction('interior', function(i)
        return not is_boundary(i)
    end)
    
    -- Should these be hidden?
    grid.cells:NewFieldMacro('xid', L.NewMacro(function(c)
        return liszt ` L.id(c) % L.addr(xsize)
    end))
    grid.cells:NewFieldMacro('yid', L.NewMacro(function(c)
        return liszt ` (L.id(c) / L.addr(xsize)) % L.addr(ysize)
    end))
    grid.cells:NewFieldMacro('zid', L.NewMacro(function(c)
        return liszt ` L.id(c) / L.addr(xysize)
    end))

    grid.cells:NewFieldMacro('center', L.NewMacro(function(c)
        return liszt ` L.vec3f({
            xorigin + xcwidth * (L.double(c.xid) + 0.5),
            yorigin + ycwidth * (L.double(c.yid) + 0.5),
            zorigin + zcwidth * (L.double(c.zid) + 0.5)
        })
    end))

    grid.cell_locate = L.NewMacro(function(xyz_vec)
        return liszt quote
            var xyz = xyz_vec -- prevent duplication
            var xval = (xyz[0] - xorigin)/xcwidth
            var yval = (xyz[1] - yorigin)/ycwidth
            var zval = (xyz[2] - zorigin)/zcwidth
            var xidx = L.addr(clamp_impl(L.int(xval), 0, xsize-1))
            var yidx = L.addr(clamp_impl(L.int(yval), 0, ysize-1))
            var yidx = L.addr(clamp_impl(L.int(zval), 0, zsize-1))
        in
            L.UNSAFE_ROW(xidx + yidx * xsize + zidx * xysize, grid.cells)
        end
    end)

    -- boundary depths
    grid.cells:NewFieldMacro('xneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(xn_bd - c.xid), 0)
    end))
    grid.cells:NewFieldMacro('xpos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(c.xid - (xsize-1 - xn_bd)), 0)
    end))
    grid.cells:NewFieldMacro('yneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(yn_bd - c.yid), 0)
    end))
    grid.cells:NewFieldMacro('ypos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(c.yid - (ysize-1 - yn_bd)), 0)
    end))
    grid.cells:NewFieldMacro('zneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(zn_bd - c.zid), 0)
    end))
    grid.cells:NewFieldMacro('zpos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(c.zid - (zsize-1 - zn_bd)), 0)
    end))

    grid.cells:NewFieldMacro('in_boundary', L.NewMacro(function(c)
        return liszt ` c.xneg_depth > 0 or c.xpos_depth > 0 or
                       c.yneg_depth > 0 or c.ypos_depth > 0 or
                       c.zneg_depth > 0 or c.zpos_depth > 0
    end))
    grid.cells:NewFieldMacro('in_interior', L.NewMacro(function(c)
        return liszt ` not c.in_boundary
    end))
end

-- There are (N+1) x (M+1) x (L+1) dual cells for an NxMxL grid
-- unless perdiodicity...
local function setup3dDualCells(grid)
    local xpd       = grid:xUsePeriodic()
    local ypd       = grid:yUsePeriodic()
    local zpd       = grid:zUsePeriodic()
    local xsize     = grid:xSize() + (xpd and 0 or 1)
    local ysize     = grid:ySize() + (ypd and 0 or 1)
    local zsize     = grid:zSize() + (zpd and 0 or 1)
    local xysize    = xsize * ysize
    local xcwidth   = grid:xCellWidth()
    local ycwidth   = grid:yCellWidth()
    local zcwidth   = grid:zCellWidth()
    local xorigin   = grid:xOrigin()
    local yorigin   = grid:yOrigin()
    local zorigin   = grid:zOrigin()
    local xn_bd     = grid:xBoundaryDepth()
    local yn_bd     = grid:yBoundaryDepth()
    local zn_bd     = grid:zBoundaryDepth()

    -- relative offset
    grid.dual_cells:NewFieldMacro('__apply_macro',
        L.NewMacro(function(dc,xoff,yoff,zoff)
            return liszt quote
                var xp = (xoff + xsize)
                var yp = (yoff + ysize)
                var zp = (zoff + zsize)
                var xi = L.addr((xp + dc.xid) % xsize)
                var yi = L.addr((yp + dc.yid) % ysize)
                var zi = L.addr((zp + dc.zid) % zsize)
            in
                L.UNSAFE_ROW( xi + yi * xsize + zi * xysize, grid.dual_cells )
            end
        end))

    -- Should these be hidden?
    grid.dual_cells:NewFieldMacro('xid', L.NewMacro(function(dc)
        return liszt ` L.id(dc) % L.addr(xsize)
    end))
    grid.dual_cells:NewFieldMacro('yid', L.NewMacro(function(dc)
        return liszt ` (L.id(dc) / L.addr(xsize)) % L.addr(ysize)
    end))
    grid.dual_cells:NewFieldMacro('zid', L.NewMacro(function(dc)
        return liszt ` L.id(dc) / L.addr(xysize)
    end))
    if not xpd and not ypd and not zpd then
        grid.dual_cells:NewFieldMacro('center', L.NewMacro(function(dc)
            return liszt `L.vec3f({
                xorigin + xcwidth * (L.double(dc.xid)),
                yorigin + ycwidth * (L.double(dc.yid)),
                zorigin + zcwidth * (L.double(dc.zid))
            })
        end))
    end

    grid.dual_locate = L.NewMacro(function(xyz_vec)
        return liszt quote
            var xyz = xyz_vec -- prevent duplication
            var xval = (xyz[0] - xorigin)/xcwidth + 0.5
            var yval = (xyz[1] - yorigin)/ycwidth + 0.5
            var zval = (xyz[2] - zorigin)/zcwidth + 0.5
            var xidx : L.addr
            var yidx : L.addr
            var zidx : L.addr
            if xpd then
                xidx = float_to_addr_mod(xval, xsize)
            else
                xidx = clamp_impl(L.int(xval), 0, xsize-1)
            end
            if ypd then
                yidx = float_to_addr_mod(yval, ysize)
            else
                yidx = clamp_impl(L.int(yval), 0, ysize-1)
            end
            if zpd then
                zidx = float_to_addr_mod(zval, zsize)
            else
                zidx = clamp_impl(L.int(zval), 0, zsize-1)
            end
        in
            L.UNSAFE_ROW(xidx + yidx * xsize + zidx * xysize, grid.dual_cells)
        end
    end)
end

-- There are (N+1) x (M+1) x (L+1) vertices for an NxMxL grid
-- unless perdiodicity...
local function setup3dVertices(grid)
    local xpd       = grid:xUsePeriodic()
    local ypd       = grid:yUsePeriodic()
    local zpd       = grid:zUsePeriodic()
    local xsize     = grid:xSize() + (xpd and 0 or 1)
    local ysize     = grid:ySize() + (ypd and 0 or 1)
    local zsize     = grid:zSize() + (zpd and 0 or 1)
    local xysize    = xsize * ysize
    local xcwidth   = grid:xCellWidth()
    local ycwidth   = grid:yCellWidth()
    local zcwidth   = grid:zCellWidth()
    local xorigin   = grid:xOrigin()
    local yorigin   = grid:yOrigin()
    local zorigin   = grid:zOrigin()
    local xn_bd     = grid:xBoundaryDepth()
    local yn_bd     = grid:yBoundaryDepth()
    local zn_bd     = grid:zBoundaryDepth()

    -- relative offset
    grid.vertices:NewFieldMacro('__apply_macro',
        L.NewMacro(function(v,xoff,yoff)
            return liszt quote
                var xp = (xoff + xsize)
                var yp = (yoff + ysize)
                var zp = (zoff + zsize)
                var xi = L.addr((xp + v.xid) % xsize)
                var yi = L.addr((yp + v.yid) % ysize)
                var zi = L.addr((zp + v.zid) % zsize)
            in
                L.UNSAFE_ROW( xi + yi * xsize + zi * xysize, grid.vertices )
            end
        end))

    -- Should these be hidden?
    grid.dual_cells:NewFieldMacro('xid', L.NewMacro(function(v)
        return liszt ` L.id(v) % L.addr(xsize)
    end))
    grid.dual_cells:NewFieldMacro('yid', L.NewMacro(function(v)
        return liszt ` (L.id(v) / L.addr(xsize)) % L.addr(ysize)
    end))
    grid.dual_cells:NewFieldMacro('zid', L.NewMacro(function(v)
        return liszt ` L.id(v) / L.addr(xysize)
    end))
end

local function setup3dInterconnects(grid)
    local xpd       = grid:xUsePeriodic()
    local ypd       = grid:yUsePeriodic()
    local zpd       = grid:zUsePeriodic()
    local cxsize, cysize, czsize = grid:xSize(), grid:ySize(), grid:zSize()
    local cxysize   = cxsize * cysize
    local vxsize    = cxsize + (xpd and 0 or 1)
    local vysize    = cysize + (ypd and 0 or 1)
    local vzsize    = czsize + (zpd and 0 or 1)
    local vxysize   = vxsize * vysize

    grid.dual_cells:NewFieldMacro('vertex', L.NewMacro(function(dc)
        return liszt ` L.UNSAFE_ROW( L.id(dc), grid.vertices )
    end))
    grid.vertices:NewFieldMacro('dual_cell', L.NewMacro(function(v)
        return liszt ` L.UNSAFE_ROW( L.id(v), grid.dual_cells )
    end))

    grid.cells:NewFieldMacro('vertex', L.NewMacro(function(c)
        return liszt ` L.UNSAFE_ROW(
            c.xid + c.yid * vxsize + c.zid * vxysize, grid.vertices )
    end))
    grid.vertices:NewFieldMacro('cell', L.NewMacro(function(v)
        return liszt ` L.UNSAFE_ROW(
            v.xid + v.yid * cxsize + v.zid * cxysize, grid.cells )
    end))
end

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
    local function is_num(obj) return type(obj) == 'number' end
    local function is_bool(obj) return type(obj) == 'boolean' end
    local function check_params(params)
        local check =
            type(params) == 'table' and
            type(params.size) == 'table' and
            type(params.origin) == 'table' and
            is_num(params.size[1]) and is_num(params.size[2]) and
            is_num(params.size[3]) and
            is_num(params.origin[1]) and is_num(params.origin[2]) and
            is_num(params.origin[3]) and
            is_num(params.width[1]) and is_num(params.width[2]) and
            is_num(params.width[3])
        if check and params.boundary_depth then
            check = check and
                    type(params.boundary_depth) == 'table' and
                    is_num(params.boundary_depth[1]) and
                    is_num(params.boundary_depth[2]) and
                    is_num(params.boundary_depth[3])
        end
        if check and params.periodic_boundary then
            check = check and
                    type(params.periodic_boundary) == 'table' and
                    is_bool(params.periodic_boundary[1]) and
                    is_bool(params.periodic_boundary[2]) and
                    is_bool(params.periodic_boundary[3])
        end
        return check
    end
    if not check_params(params) then error(calling_convention, 2) end

    -- default
    params.periodic_boundary = params.periodic_boundary or
                                {false, false, false}
    params.boundary_depth    = params.boundary_depth or
                                {1, 1, 1}
    for i=1,3 do
        if params.periodic_boundary[i] then
            params.boundary_depth[i] = 0
        end
    end

    -- sizes
    local nCells        = params.size[1] * params.size[2] * params.size[3]
    local dcsize        = {}
    local nDualCells    = 1
    for i=1,3 do
        if params.periodic_boundary[i] then
            dcsize[i] = params.size[i]
        else
            dcsize[i] = params.size[i] + 1
        end
        nDualCells = nDualCells * dcsize[i]
    end
    local nVerts        = nDualCells

    local grid = setmetatable({
        _n_xyz      = params.size,
        _origin     = params.origin,
        _dims       = params.width,
        _bd_depth   = params.boundary_depth,
        _periodic   = params.periodic_boundary,
        -- relations
        cells       = L.NewRelation(nCells, 'cells'),
        dual_cells  = L.NewRelation(nDualCells, 'dual_cells'),
        vertices    = L.NewRelation(nVerts, 'vertices'),
    }, Grid2d)

    setup3dCells(grid)
    setup3dDualCells(grid)
    setup3dVertices(grid)
    setup3dInterconnects(grid)

    return grid
end



function Grid3d:xSize()             return self._n_xy[1]            end
function Grid3d:ySize()             return self._n_xy[2]            end
function Grid3d:zSize()             return self._n_xy[3]            end
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






