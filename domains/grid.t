import "compiler.liszt"

local Grid = {}
package.loaded["domains.grid"] = Grid

local L = require "compiler.lisztlib"

local Grid2d = {}
local Grid3d = {}
Grid2d.__index = Grid2d
Grid3d.__index = Grid3d

local int_floor = L.NewMacro(function(v)
    return liszt ` L.int(L.floor(v))
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

local function copy_table(tbl)
    local cpy = {}
    for k,v in pairs(tbl) do cpy[k] = v end
    return cpy
end


-- There are N x M cells for an NxM grid
local function setup2dCells(grid)
    local xsize, ysize      = grid:xSize(), grid:ySize()
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
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
            xorigin + xcwidth * (L.double(c.xid) + 0.5),
            yorigin + ycwidth * (L.double(c.yid) + 0.5)
        })
    end))

    grid.cell_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xy = xy_vec -- prevent duplication
            var xval = (xy[0] - xorigin)/xcwidth
            var yval = (xy[1] - yorigin)/ycwidth
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
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local xsize, ysize      = grid._vn_xy[1], grid._vn_xy[2]
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

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
                xorigin + xcwidth * (L.double(dc.xid)),
                yorigin + ycwidth * (L.double(dc.yid))
            })
        end))
    end

    grid.dual_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xy = xy_vec -- prevent duplication
            var xval = (xy[0] - xorigin)/xcwidth + 0.5
            var yval = (xy[1] - yorigin)/ycwidth + 0.5
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
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local xsize, ysize      = grid._vn_xy[1], grid._vn_xy[2]
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

    -- relative offset
    grid.vertices:NewFieldMacro('__apply_macro',
        L.NewMacro(function(v,xoff,yoff)
            return liszt quote
                var xp = (xoff + xsize)
                var yp = (yoff + ysize)
                var xi = L.addr((xp + v.xid) % xsize)
                var yi = L.addr((yp + v.yid) % ysize)
            in
                L.UNSAFE_ROW( xi + yi * xsize, grid.vertices )
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
    grid.vertices:NewSubsetFromFunction('boundary', is_boundary)
    grid.vertices:NewSubsetFromFunction('interior', function(i)
        return not is_boundary(i)
    end)

    -- Should these be hidden?
    grid.vertices:NewFieldMacro('xid', L.NewMacro(function(v)
        return liszt ` L.id(v) % L.addr(xsize)
    end))
    grid.vertices:NewFieldMacro('yid', L.NewMacro(function(v)
        return liszt ` L.id(v) / L.addr(xsize)
    end))

    -- boundary depths
    grid.vertices:NewFieldMacro('xneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(xn_bd - v.xid), 0)
    end))
    grid.vertices:NewFieldMacro('xpos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(v.xid - (xsize-1 - xn_bd)), 0)
    end))
    grid.vertices:NewFieldMacro('yneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(yn_bd - v.yid), 0)
    end))
    grid.vertices:NewFieldMacro('ypos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(v.yid - (ysize-1 - yn_bd)), 0)
    end))

    grid.vertices:NewFieldMacro('in_boundary', L.NewMacro(function(v)
        return liszt ` v.xneg_depth > 0 or v.xpos_depth > 0 or
                       v.yneg_depth > 0 or v.ypos_depth > 0
    end))
    grid.vertices:NewFieldMacro('in_interior', L.NewMacro(function(v)
        return liszt ` not v.in_boundary
    end))
end

local function setup2dEdges(grid)
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local cxsize, cysize    = grid:xSize(), grid:ySize()
    local vxsize, vysize    = grid._vn_xy[1], grid._vn_xy[2]
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

    -- weird edge value stuff
    local nxedge            = cxsize * vysize
    local nyedge            = vxsize * cysize
    local xedge_off         = 0
    local yedge_off         = nxedge

    grid.edges:NewFieldMacro('is_x', L.NewMacro(function(e)
        return liszt ` L.id(e) < yedge_off
    end))
    grid.edges:NewFieldMacro('is_y', L.NewMacro(function(e)
        return liszt ` L.id(e) >= yedge_off
    end))

    -- subset versions
    grid.edges:NewSubsetFromFunction('x_aligned', function(i)
        return i < yedge_off
    end)
    grid.edges:NewSubsetFromFunction('y_aligned', function(i)
        return i >= yedge_off
    end)


    -- in the case this is an x-aligned edge
    grid.edges:NewFieldMacro('x_xid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - xedge_off) % L.addr(cxsize)
    end))
    grid.edges:NewFieldMacro('x_yid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - xedge_off) / L.addr(cxsize)
    end))
    -- in the case this is a y-aligned edge
    grid.edges:NewFieldMacro('y_xid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - yedge_off) % L.addr(vxsize)
    end))
    grid.edges:NewFieldMacro('y_yid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - yedge_off) / L.addr(vxsize)
    end))

    -- relative offset
    grid.edges:NewFieldMacro('__apply_macro',
        L.NewMacro(function(e,xoff,yoff)
            return liszt quote
                var result : L.addr
                if e.is_x then
                    var xsize = cxsize
                    var ysize = vysize
                    var xi = L.addr((xoff + xsize + e.x_xid) % xsize)
                    var yi = L.addr((yoff + ysize + e.x_yid) % ysize)
                    result = xedge_off + xi + yi * xsize
                else
                    var xsize = vxsize
                    var ysize = cysize
                    var xi = L.addr((xoff + xsize + e.y_xid) % xsize)
                    var yi = L.addr((yoff + ysize + e.y_yid) % ysize)
                    result = yedge_off + xi + yi * xsize
                end
            in
                L.UNSAFE_ROW(result, grid.edges)
            end
        end))
end

local function setup2dInterconnects(grid)
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local cxsize, cysize    = grid:xSize(), grid:ySize()
    local vxsize, vysize    = grid._vn_xy[1], grid._vn_xy[2]

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

    -- edge connects
    local nxedge            = cxsize * vysize
    local nyedge            = vxsize * cysize
    local xedge_off         = 0
    local yedge_off         = nxedge

    grid.edges:NewFieldMacro('tail', L.NewMacro(function(e)
        return liszt quote
            var result : L.addr
            if e.is_x then
                result = e.x_xid + e.x_yid * vxsize
            else
                result = e.y_xid + e.y_yid * vxsize
            end
        in
            L.UNSAFE_ROW( result, grid.vertices )
        end
    end))
    -- just offset the tail
    grid.edges:NewFieldMacro('head', L.NewMacro(function(e)
        return liszt quote
            var tail = e.tail
            var v : L.addr
            if e.is_x then  v = L.id(tail(1,0))
            else            v = L.id(tail(0,1))
            end
        in
            L.UNSAFE_ROW( v, grid.vertices )
        end
    end))
    grid.vertices:NewFieldMacro('xedge', L.NewMacro(function(v)
        return liszt `
            L.UNSAFE_ROW( xedge_off + v.xid + v.yid * cxsize, grid.edges )
    end))
    grid.vertices:NewFieldMacro('yedge', L.NewMacro(function(v)
        return liszt `
            L.UNSAFE_ROW( yedge_off + v.xid + v.yid * vxsize, grid.edges )
    end))
end

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
    local function is_num(obj) return type(obj) == 'number' end
    local function is_bool(obj) return type(obj) == 'boolean' end
    local function check_params(params)
        local check =
            type(params) == 'table' and
            type(params.size) == 'table' and
            type(params.origin) == 'table' and
            is_num(params.size[1]) and is_num(params.size[2]) and
            is_num(params.origin[1]) and is_num(params.origin[2]) and
            is_num(params.width[1]) and is_num(params.width[2])
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
    local vsize              = copy_table(params.size)
    for i=1,2 do
        if params.periodic_boundary[i] then
            params.boundary_depth[i] = 0
        else
            vsize[i] = vsize[i] + 1
        end
    end

    local nCells        = params.size[1] * params.size[2]
    local nVerts        = vsize[1] * vsize[2]
    local nDualCells    = nVerts
    local nEdges        = params.size[1] * vsize[2]
                        + vsize[1] * params.size[2]

    local grid = setmetatable({
        _vn_xy      = vsize, -- just for internal use, not exposed
        _n_xy       = copy_table(params.size),
        _origin     = copy_table(params.origin),
        _dims       = copy_table(params.width),
        _bd_depth   = copy_table(params.boundary_depth),
        _periodic   = copy_table(params.periodic_boundary),
        -- relations
        cells       = L.NewRelation { size = nCells, name = 'cells' },
        edges       = L.NewRelation { size = nEdges, name = 'edges' },
        dual_cells  = L.NewRelation { size = nDualCells, name = 'dual_cells' },
        vertices    = L.NewRelation { size = nVerts, name = 'vertices' },
    }, Grid2d)

    setup2dCells(grid)
    setup2dDualCells(grid)
    setup2dVertices(grid)
    setup2dEdges(grid)
    setup2dInterconnects(grid)

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
            var zidx = L.addr(clamp_impl(L.int(zval), 0, zsize-1))
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
    local xsize     = grid._vn_xyz[1]
    local ysize     = grid._vn_xyz[2]
    local zsize     = grid._vn_xyz[3]
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
        L.NewMacro(function(v,xoff,yoff, zoff)
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
    grid.vertices:NewSubsetFromFunction('boundary', is_boundary)
    grid.vertices:NewSubsetFromFunction('interior', function(i)
        return not is_boundary(i)
    end)

    -- Should these be hidden?
    grid.vertices:NewFieldMacro('xid', L.NewMacro(function(v)
        return liszt ` L.id(v) % L.addr(xsize)
    end))
    grid.vertices:NewFieldMacro('yid', L.NewMacro(function(v)
        return liszt ` (L.id(v) / L.addr(xsize)) % L.addr(ysize)
    end))
    grid.vertices:NewFieldMacro('zid', L.NewMacro(function(v)
        return liszt ` L.id(v) / L.addr(xysize)
    end))

    -- boundary depths
    grid.vertices:NewFieldMacro('xneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(xn_bd - v.xid), 0)
    end))
    grid.vertices:NewFieldMacro('xpos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(v.xid - (xsize-1 - xn_bd)), 0)
    end))
    grid.vertices:NewFieldMacro('yneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(yn_bd - v.yid), 0)
    end))
    grid.vertices:NewFieldMacro('ypos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(v.yid - (ysize-1 - yn_bd)), 0)
    end))
    grid.vertices:NewFieldMacro('zneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(zn_bd - v.zid), 0)
    end))
    grid.vertices:NewFieldMacro('zpos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(v.zid - (zsize-1 - zn_bd)), 0)
    end))


    grid.vertices:NewFieldMacro('in_boundary', L.NewMacro(function(v)
        return liszt ` v.xneg_depth > 0 or v.xpos_depth > 0 or
                       v.yneg_depth > 0 or v.ypos_depth > 0 or
                       v.zneg_depth > 0 or v.zpos_depth > 0
    end))
    grid.vertices:NewFieldMacro('in_interior', L.NewMacro(function(v)
        return liszt ` not v.in_boundary
    end))
end

local function setup3dEdges(grid)
    local xpd       = grid:xUsePeriodic()
    local ypd       = grid:yUsePeriodic()
    local zpd       = grid:zUsePeriodic()
    local cxsize    = grid:xSize()
    local cysize    = grid:ySize()
    local czsize    = grid:zSize()
    local vxsize    = grid._vn_xyz[1]
    local vysize    = grid._vn_xyz[2]
    local vzsize    = grid._vn_xyz[3]
    local xcwidth   = grid:xCellWidth()
    local ycwidth   = grid:yCellWidth()
    local zcwidth   = grid:zCellWidth()
    local xorigin   = grid:xOrigin()
    local yorigin   = grid:yOrigin()
    local zorigin   = grid:zOrigin()
    local xn_bd     = grid:xBoundaryDepth()
    local yn_bd     = grid:yBoundaryDepth()
    local zn_bd     = grid:zBoundaryDepth()

    -- weird edge value stuff
    local nxedge            = cxsize * vysize * vzsize
    local nyedge            = vxsize * cysize * vzsize
    local nzedge            = vxsize * vysize * czsize
    local xedge_off         = 0
    local yedge_off         = nxedge
    local zedge_off         = yedge_off + nyedge

    grid.edges:NewFieldMacro('is_x', L.NewMacro(function(e)
        return liszt ` L.id(e) < yedge_off
    end))
    grid.edges:NewFieldMacro('is_y', L.NewMacro(function(e)
        return liszt ` L.id(e) >= yedge_off and L.id(e) < zedge_off
    end))
    grid.edges:NewFieldMacro('is_z', L.NewMacro(function(e)
        return liszt ` L.id(e) >= zedge_off
    end))

    -- subset versions
    grid.edges:NewSubsetFromFunction('x_aligned', function(i)
        return i < yedge_off
    end)
    grid.edges:NewSubsetFromFunction('y_aligned', function(i)
        return i >= yedge_off and i < zedge_off
    end)

    grid.edges:NewSubsetFromFunction('z_aligned', function(i)
        return i >= zedge_off
    end)

    -- in the case this is an x-aligned edge
    grid.edges:NewFieldMacro('x_xid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - xedge_off) % L.addr(cxsize)
    end))
    grid.edges:NewFieldMacro('x_yid', L.NewMacro(function(e)
        return liszt `
            ((L.id(e) - xedge_off) / L.addr(cxsize)) % L.addr(vysize)
    end))
    grid.edges:NewFieldMacro('x_zid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - xedge_off) / L.addr(cxsize * vysize)
    end))
    -- in the case this is a y-aligned edge
    grid.edges:NewFieldMacro('y_xid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - yedge_off) % L.addr(vxsize)
    end))
    grid.edges:NewFieldMacro('y_yid', L.NewMacro(function(e)
        return liszt `
            ((L.id(e) - yedge_off) / L.addr(vxsize)) % L.addr(cysize)
    end))
    grid.edges:NewFieldMacro('y_zid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - yedge_off) / L.addr(vxsize * cysize)
    end))
    -- in the case this is a z-aligned edge
    grid.edges:NewFieldMacro('z_xid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - zedge_off) % L.addr(vxsize)
    end))
    grid.edges:NewFieldMacro('z_yid', L.NewMacro(function(e)
        return liszt `
            ((L.id(e) - zedge_off) / L.addr(vxsize)) % L.addr(vysize)
    end))
    grid.edges:NewFieldMacro('z_zid', L.NewMacro(function(e)
        return liszt ` (L.id(e) - zedge_off) / L.addr(vxsize * vysize)
    end))

    -- relative offset
    grid.edges:NewFieldMacro('__apply_macro',
      L.NewMacro(function(e,xoff,yoff,zoff)
        return liszt quote
            var xsize : L.addr = vxsize
            var ysize : L.addr = vysize
            var zsize : L.addr = vzsize
            var offset : L.addr
            var xi : L.addr
            var yi : L.addr
            var zi : L.addr

            if e.is_x then
                xsize  = cxsize
                offset = xedge_off
                xi = L.addr((xoff + xsize + e.x_xid) % xsize)
                yi = L.addr((yoff + ysize + e.x_yid) % ysize)
                zi = L.addr((zoff + zsize + e.x_zid) % zsize)
            elseif e.is_y then
                ysize  = cysize
                offset = yedge_off
                xi = L.addr((xoff + xsize + e.y_xid) % xsize)
                yi = L.addr((yoff + ysize + e.y_yid) % ysize)
                zi = L.addr((zoff + zsize + e.y_zid) % zsize)
            else
                zsize  = czsize
                offset = zedge_off
                xi = L.addr((xoff + xsize + e.z_xid) % xsize)
                yi = L.addr((yoff + ysize + e.z_yid) % ysize)
                zi = L.addr((zoff + zsize + e.z_zid) % zsize)
            end
            var xysize = xsize * ysize
        in
            L.UNSAFE_ROW( offset + xi + yi * xsize + zi * xysize,
                          grid.edges )
        end
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

    -- edge connects
    local ex_xysize         = cxsize * vysize
    local ey_xysize         = vxsize * cysize
    local ez_xysize         = vxsize * vysize
    local nxedge            = cxsize * vysize * vzsize
    local nyedge            = vxsize * cysize * vzsize
    local nzedge            = vxsize * vysize * czsize
    local xedge_off         = 0
    local yedge_off         = nxedge
    local zedge_off         = yedge_off + nyedge

    grid.edges:NewFieldMacro('tail', L.NewMacro(function(e)
        return liszt quote
            var result : L.addr
            if e.is_x then
                result = e.x_xid + e.x_yid * vxsize + e.x_zid * vxysize
            elseif e.is_y then
                result = e.y_xid + e.y_yid * vxsize + e.y_zid * vxysize
            else
                result = e.z_xid + e.z_yid * vxsize + e.z_zid * vxysize
            end
        in
            L.UNSAFE_ROW( result, grid.vertices )
        end
    end))
    -- just offset the tail
    grid.edges:NewFieldMacro('head', L.NewMacro(function(e)
        return liszt quote
            var tail = e.tail
            var v : L.addr
                if e.is_x then  v = L.id(tail(1,0,0))
            elseif e.is_y then  v = L.id(tail(0,1,0))
            else                v = L.id(tail(0,0,1))
            end
        in
            L.UNSAFE_ROW( v, grid.vertices )
        end
    end))
    grid.vertices:NewFieldMacro('xedge', L.NewMacro(function(v)
        return liszt ` L.UNSAFE_ROW(
            xedge_off + v.xid + (v.yid + v.zid * vysize) * cxsize, grid.edges )
    end))
    grid.vertices:NewFieldMacro('yedge', L.NewMacro(function(v)
        return liszt ` L.UNSAFE_ROW(
            yedge_off + v.xid + (v.yid + v.zid * cysize) * vxsize, grid.edges )
    end))
    grid.vertices:NewFieldMacro('zedge', L.NewMacro(function(v)
        return liszt ` L.UNSAFE_ROW(
            zedge_off + v.xid + (v.yid + v.zid * vysize) * vxsize, grid.edges )
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
    params.boundary_depth    = params.boundary_depth or {1, 1, 1}
    local vsize              = copy_table(params.size)
    for i=1,3 do
        if params.periodic_boundary[i] then
            params.boundary_depth[i] = 0
        else
            vsize[i] = vsize[i] + 1
        end
    end

    -- sizes
    local nCells        = params.size[1] * params.size[2] * params.size[3]
    local nVerts        = vsize[1] * vsize[2] * vsize[3]
    local nDualCells    = nVerts
    local nEdges        = params.size[1] * vsize[2] * vsize[3]
                        + vsize[1] * params.size[2] * vsize[3]
                        + vsize[1] * vsize[2] * params.size[3]

    local grid = setmetatable({
        _vn_xyz     = vsize, -- just for internal use, not exposed
        _n_xyz      = copy_table(params.size),
        _origin     = copy_table(params.origin),
        _dims       = copy_table(params.width),
        _bd_depth   = copy_table(params.boundary_depth),
        _periodic   = copy_table(params.periodic_boundary),
        -- relations
        cells       = L.NewRelation { size = nCells, name = 'cells' },
        edges       = L.NewRelation { size = nEdges, name = 'edges' },
        dual_cells  = L.NewRelation { size = nDualCells, name = 'dual_cells' },
        vertices    = L.NewRelation { size = nVerts, name = 'vertices' },
    }, Grid3d)

    setup3dCells(grid)
    setup3dDualCells(grid)
    setup3dVertices(grid)
    setup3dEdges(grid)
    setup3dInterconnects(grid)

    return grid
end



function Grid3d:xSize()             return self._n_xyz[1]            end
function Grid3d:ySize()             return self._n_xyz[2]            end
function Grid3d:zSize()             return self._n_xyz[3]            end
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






