import "ebb.liszt"

local Grid = {}
package.loaded["ebb.domains.grid"] = Grid

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
local float_to_uint64_mod = L.NewMacro(function(x, m)
    return liszt `( int_floor(x) % m + m ) % m
end)

local function copy_table(tbl)
    local cpy = {}
    for k,v in pairs(tbl) do cpy[k] = v end
    return cpy
end


local function setup2dCells(grid)
    local xsize, ysize      = grid:xSize(), grid:ySize()
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

    grid.cells:NewFieldMacro('__apply_macro',
        L.NewMacro(function(c,xoff,yoff)
            return liszt `L.Affine(grid.cells, {{1,0,xoff},
                                                {0,1,yoff}}, c)
        end))

    -- Boundary/Interior subsets
    local function is_boundary(x,y)
        return y <  yn_bd or y >= ysize-yn_bd or
               x <  xn_bd or x >= xsize-xn_bd
    end
    grid.cells:NewSubsetFromFunction('boundary', is_boundary)
    grid.cells:NewSubsetFromFunction('interior', function(xi,yi)
        return not is_boundary(xi,yi)
    end)

    grid.cells:NewFieldMacro('center', L.NewMacro(function(c)
        return liszt ` L.vec2f({
            xorigin + xcwidth * (L.double(L.xid(c)) + 0.5),
            yorigin + ycwidth * (L.double(L.yid(c)) + 0.5)
        })
    end))

    grid.cell_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xy = xy_vec -- prevent duplication
            var xval = (xy[0] - xorigin)/xcwidth
            var yval = (xy[1] - yorigin)/ycwidth
            var xidx = L.uint64(clamp_impl(L.int(xval), 0, xsize-1))
            var yidx = L.uint64(clamp_impl(L.int(yval), 0, ysize-1))
        in
            L.UNSAFE_ROW({xidx, yidx}, grid.cells)
        end
    end)

    -- boundary depths
    grid.cells:NewFieldMacro('xneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(xn_bd - L.xid(c)), 0)
    end))
    grid.cells:NewFieldMacro('xpos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(L.xid(c) - (xsize-1 - xn_bd)), 0)
    end))
    grid.cells:NewFieldMacro('yneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(yn_bd - L.yid(c)), 0)
    end))
    grid.cells:NewFieldMacro('ypos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(L.yid(c) - (ysize-1 - yn_bd)), 0)
    end))

    grid.cells:NewFieldMacro('in_boundary', L.NewMacro(function(c)
        return liszt ` c.xneg_depth > 0 or c.xpos_depth > 0 or
                       c.yneg_depth > 0 or c.ypos_depth > 0
    end))
    grid.cells:NewFieldMacro('in_interior', L.NewMacro(function(c)
        return liszt ` not c.in_boundary
    end))
end

local function setup2dDualCells(grid)
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local xsize, ysize      = grid._vn_xy[1], grid._vn_xy[2]
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

    grid.dual_cells:NewFieldMacro('__apply_macro',
        L.NewMacro(function(dc,xoff,yoff)
            return liszt `L.Affine(grid.dual_cells, {{1,0,xoff},
                                                     {0,1,yoff}}, dc)
        end))

    if not xpd and not ypd then
        grid.dual_cells:NewFieldMacro('center', L.NewMacro(function(dc)
            return liszt `L.vec2f({
                xorigin + xcwidth * (L.double(L.xid(dc))),
                yorigin + ycwidth * (L.double(L.yid(dc)))
            })
        end))
    end

    grid.dual_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xy = xy_vec -- prevent duplication
            var xval = (xy[0] - xorigin)/xcwidth + 0.5
            var yval = (xy[1] - yorigin)/ycwidth + 0.5
            var xidx : L.uint64
            var yidx : L.uint64
            if xpd then
                xidx = float_to_uint64_mod(xval, xsize)
            else
                xidx = clamp_impl(L.int(xval), 0, xsize-1)
            end
            if ypd then
                yidx = float_to_uint64_mod(yval, ysize)
            else
                yidx = clamp_impl(L.int(yval), 0, ysize-1)
            end
        in
            L.UNSAFE_ROW({xidx, yidx}, grid.dual_cells)
        end
    end)
end

local function setup2dVertices(grid)
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local xsize, ysize      = grid._vn_xy[1], grid._vn_xy[2]
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

    grid.vertices:NewFieldMacro('__apply_macro',
        L.NewMacro(function(v,xoff,yoff)
            return liszt `L.Affine(grid.vertices, {{1,0,xoff},
                                                   {0,1,yoff}}, v)
        end))

    -- Boundary/Interior subsets
    local function is_boundary(x,y)
        return y < yn_bd or y >= ysize-yn_bd or
               x < xn_bd or x >= xsize-xn_bd
    end
    grid.vertices:NewSubsetFromFunction('boundary', is_boundary)
    grid.vertices:NewSubsetFromFunction('interior', function(xi,yi)
        return not is_boundary(xi,yi)
    end)

    -- boundary depths
    grid.vertices:NewFieldMacro('xneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(xn_bd - L.xid(v)), 0)
    end))
    grid.vertices:NewFieldMacro('xpos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(L.xid(v) - (xsize-1 - xn_bd)), 0)
    end))
    grid.vertices:NewFieldMacro('yneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(yn_bd - L.yid(v)), 0)
    end))
    grid.vertices:NewFieldMacro('ypos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(L.yid(v) - (ysize-1 - yn_bd)), 0)
    end))

    grid.vertices:NewFieldMacro('in_boundary', L.NewMacro(function(v)
        return liszt ` v.xneg_depth > 0 or v.xpos_depth > 0 or
                       v.yneg_depth > 0 or v.ypos_depth > 0
    end))
    grid.vertices:NewFieldMacro('in_interior', L.NewMacro(function(v)
        return liszt ` not v.in_boundary
    end))
end
local function setup2dDualVertices(grid)
    local xsize, ysize      = grid:xSize(), grid:ySize()
    local xcwidth, ycwidth  = grid:xCellWidth(), grid:yCellWidth()
    local xorigin, yorigin  = grid:xOrigin(), grid:yOrigin()
    local xn_bd             = grid:xBoundaryDepth()
    local yn_bd             = grid:yBoundaryDepth()

    grid.dual_vertices:NewFieldMacro('__apply_macro',
        L.NewMacro(function(dv,xoff,yoff)
            return liszt `L.Affine(grid.dual_vertices, {{1,0,xoff},
                                                        {0,1,yoff}}, dv)
        end))
end

local function setup2dInterconnects(grid)
    local xpd, ypd          = grid:xUsePeriodic(), grid:yUsePeriodic()
    local cxsize, cysize    = grid:xSize(), grid:ySize()
    local vxsize, vysize    = grid._vn_xy[1], grid._vn_xy[2]

    -- v <-> dc
    grid.dual_cells:NewFieldMacro('vertex',
        L.NewMacro(function(dc)
            return liszt `L.Affine(grid.vertices, {{1,0,0},
                                                   {0,1,0}}, dc)
        end))
    grid.vertices:NewFieldMacro('dual_cell',
        L.NewMacro(function(v)
            return liszt `L.Affine(grid.dual_cells, {{1,0,0},
                                                     {0,1,0}}, v)
        end))

    -- v <-> c
    grid.cells:NewFieldMacro('vertex',
        L.NewMacro(function(c)
            return liszt `L.Affine(grid.vertices, {{1,0,0},
                                                   {0,1,0}}, c)
        end))
    grid.vertices:NewFieldMacro('cell',
        L.NewMacro(function(v)
            return liszt `L.Affine(grid.cells, {{1,0,0},
                                                {0,1,0}}, v)
        end))

    -- dv <-> c
    grid.cells:NewFieldMacro('dual_vertex',
        L.NewMacro(function(c)
            return liszt `L.Affine(grid.dual_vertices, {{1,0,0},
                                                        {0,1,0}}, c)
        end))
    grid.dual_vertices:NewFieldMacro('cell',
        L.NewMacro(function(dv)
            return liszt `L.Affine(grid.cells, {{1,0,0},
                                                {0,1,0}}, dv)
        end))

    -- dv <-> dc
    grid.dual_cells:NewFieldMacro('dual_vertex',
        L.NewMacro(function(dc)
            return liszt `L.Affine(grid.dual_vertices, {{1,0,0},
                                                        {0,1,0}}, dc)
        end))
    grid.dual_vertices:NewFieldMacro('dual_cell',
        L.NewMacro(function(dv)
            return liszt `L.Affine(grid.dual_cells, {{1,0,0},
                                                     {0,1,0}}, dv)
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
    local wrap_bnd      = params.periodic_boundary or {false, false}
    local bnd_depth     = params.boundary_depth    or {1, 1}
    local vsize         = copy_table(params.size)
    for i=1,2 do
        if wrap_bnd[i] then bnd_depth[i] = 0
                       else vsize[i] = vsize[i] + 1 end
    end

    local grid = setmetatable({
        _vn_xy      = vsize, -- just for internal use, not exposed
        _n_xy       = copy_table(params.size),
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
    }, Grid2d)

    setup2dCells(grid)
    setup2dDualCells(grid)
    setup2dVertices(grid)
    setup2dDualVertices(grid)
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
            return liszt `L.Affine(grid.cells, {{1,0,0,xoff},
                                                {0,1,0,yoff},
                                                {0,0,1,zoff}}, c)
        end))

    -- Boundary/Interior subsets
    local function is_boundary(xi, yi, zi)
        return  xi < xn_bd or xi >= xsize-xn_bd or
                yi < yn_bd or yi >= ysize-yn_bd or
                zi < zn_bd or zi >= zsize-zn_bd
    end
    grid.cells:NewSubsetFromFunction('boundary', is_boundary)
    grid.cells:NewSubsetFromFunction('interior', function(xi,yi,zi)
        return not is_boundary(xi,yi,zi)
    end)

    grid.cells:NewFieldMacro('center', L.NewMacro(function(c)
        return liszt ` L.vec3f({
            xorigin + xcwidth * (L.double(L.xid(c)) + 0.5),
            yorigin + ycwidth * (L.double(L.yid(c)) + 0.5),
            zorigin + zcwidth * (L.double(L.zid(c)) + 0.5)
        })
    end))

    grid.cell_locate = L.NewMacro(function(xyz_vec)
        return liszt quote
            var xyz = xyz_vec -- prevent duplication
            var xval = (xyz[0] - xorigin)/xcwidth
            var yval = (xyz[1] - yorigin)/ycwidth
            var zval = (xyz[2] - zorigin)/zcwidth
            var xidx = L.uint64(clamp_impl(L.int(xval), 0, xsize-1))
            var yidx = L.uint64(clamp_impl(L.int(yval), 0, ysize-1))
            var zidx = L.uint64(clamp_impl(L.int(zval), 0, zsize-1))
        in
            L.UNSAFE_ROW({xidx, yidx, zidx}, grid.cells)
        end
    end)

    -- boundary depths
    grid.cells:NewFieldMacro('xneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(xn_bd - L.xid(c)), 0)
    end))
    grid.cells:NewFieldMacro('xpos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(L.xid(c) - (xsize-1 - xn_bd)), 0)
    end))
    grid.cells:NewFieldMacro('yneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(yn_bd - L.yid(c)), 0)
    end))
    grid.cells:NewFieldMacro('ypos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(L.yid(c) - (ysize-1 - yn_bd)), 0)
    end))
    grid.cells:NewFieldMacro('zneg_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(zn_bd - L.zid(c)), 0)
    end))
    grid.cells:NewFieldMacro('zpos_depth', L.NewMacro(function(c)
        return liszt `max_impl(L.int(L.zid(c) - (zsize-1 - zn_bd)), 0)
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
            return liszt `L.Affine(grid.dual_cells, {{1,0,0,xoff},
                                                     {0,1,0,yoff},
                                                     {0,0,1,zoff}}, dc)
        end))

    if not xpd and not ypd and not zpd then
        grid.dual_cells:NewFieldMacro('center', L.NewMacro(function(dc)
            return liszt `L.vec3f({
                xorigin + xcwidth * (L.double(L.xid(dc))),
                yorigin + ycwidth * (L.double(L.yid(dc))),
                zorigin + zcwidth * (L.double(L.zid(dc)))
            })
        end))
    end

    grid.dual_locate = L.NewMacro(function(xyz_vec)
        return liszt quote
            var xyz = xyz_vec -- prevent duplication
            var xval = (xyz[0] - xorigin)/xcwidth + 0.5
            var yval = (xyz[1] - yorigin)/ycwidth + 0.5
            var zval = (xyz[2] - zorigin)/zcwidth + 0.5
            var xidx : L.uint64
            var yidx : L.uint64
            var zidx : L.uint64
            if xpd then
                xidx = float_to_uint64_mod(xval, xsize)
            else
                xidx = clamp_impl(L.int(xval), 0, xsize-1)
            end
            if ypd then
                yidx = float_to_uint64_mod(yval, ysize)
            else
                yidx = clamp_impl(L.int(yval), 0, ysize-1)
            end
            if zpd then
                zidx = float_to_uint64_mod(zval, zsize)
            else
                zidx = clamp_impl(L.int(zval), 0, zsize-1)
            end
        in
            L.UNSAFE_ROW({xidx, yidx, zidx}, grid.dual_cells)
        end
    end)
end

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
        L.NewMacro(function(v,xoff,yoff,zoff)
            return liszt `L.Affine(grid.vertices, {{1,0,0,xoff},
                                                   {0,1,0,yoff},
                                                   {0,0,1,zoff}}, v)
        end))

    -- Boundary/Interior subsets
    local function is_boundary(xi,yi,zi)
        return  xi < xn_bd or xi >= xsize-xn_bd or
                yi < yn_bd or yi >= ysize-yn_bd or
                zi < zn_bd or zi >= zsize-zn_bd
    end
    grid.vertices:NewSubsetFromFunction('boundary', is_boundary)
    grid.vertices:NewSubsetFromFunction('interior', function(xi,yi,zi)
        return not is_boundary(xi,yi,zi)
    end)

    -- boundary depths
    grid.vertices:NewFieldMacro('xneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(xn_bd - L.xid(v)), 0)
    end))
    grid.vertices:NewFieldMacro('xpos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(L.xid(v) - (xsize-1 - xn_bd)), 0)
    end))
    grid.vertices:NewFieldMacro('yneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(yn_bd - L.yid(v)), 0)
    end))
    grid.vertices:NewFieldMacro('ypos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(L.yid(v) - (ysize-1 - yn_bd)), 0)
    end))
    grid.vertices:NewFieldMacro('zneg_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(zn_bd - L.zid(v)), 0)
    end))
    grid.vertices:NewFieldMacro('zpos_depth', L.NewMacro(function(v)
        return liszt `max_impl(L.int(L.zid(v) - (zsize-1 - zn_bd)), 0)
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

local function setup3dDualVertices(grid)

    grid.dual_vertices:NewFieldMacro('__apply_macro',
        L.NewMacro(function(dv,xoff,yoff,zoff)
            return liszt `L.Affine(grid.dual_vertices, {{1,0,0,xoff},
                                                        {0,1,0,yoff},
                                                        {0,0,1,zoff}}, dv)
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

    -- v <-> dc
    grid.dual_cells:NewFieldMacro('vertex',
        L.NewMacro(function(dc)
            return liszt `L.Affine(grid.vertices, {{1,0,0,0},
                                                   {0,1,0,0},
                                                   {0,0,1,0}}, dc)
        end))
    grid.vertices:NewFieldMacro('dual_cell',
        L.NewMacro(function(v)
            return liszt `L.Affine(grid.dual_cells, {{1,0,0,0},
                                                     {0,1,0,0},
                                                     {0,0,1,0}}, v)
        end))

    -- v <-> c
    grid.cells:NewFieldMacro('vertex',
        L.NewMacro(function(c)
            return liszt `L.Affine(grid.vertices, {{1,0,0,0},
                                                   {0,1,0,0},
                                                   {0,0,1,0}}, c)
        end))
    grid.vertices:NewFieldMacro('cell',
        L.NewMacro(function(v)
            return liszt `L.Affine(grid.cells, {{1,0,0,0},
                                                {0,1,0,0},
                                                {0,0,1,0}}, v)
        end))

    -- dv <-> c
    grid.cells:NewFieldMacro('dual_vertex',
        L.NewMacro(function(c)
            return liszt `L.Affine(grid.dual_vertices, {{1,0,0,0},
                                                        {0,1,0,0},
                                                        {0,0,1,0}}, c)
        end))
    grid.dual_vertices:NewFieldMacro('cell',
        L.NewMacro(function(dv)
            return liszt `L.Affine(grid.cells, {{1,0,0,0},
                                                {0,1,0,0},
                                                {0,0,1,0}}, dv)
        end))

    -- dv <-> dc
    grid.dual_cells:NewFieldMacro('dual_vertex',
        L.NewMacro(function(dc)
            return liszt `L.Affine(grid.dual_vertices, {{1,0,0,0},
                                                        {0,1,0,0},
                                                        {0,0,1,0}}, dc)
        end))
    grid.dual_vertices:NewFieldMacro('dual_cell',
        L.NewMacro(function(dv)
            return liszt `L.Affine(grid.dual_cells, {{1,0,0,0},
                                                     {0,1,0,0},
                                                     {0,0,1,0}}, dv)
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
    local wrap_bnd      = params.periodic_boundary or {false, false, false}
    local bnd_depth     = params.boundary_depth    or {1, 1, 1}
    local vsize         = copy_table(params.size)
    for i=1,3 do
        if wrap_bnd[i] then bnd_depth[i] = 0
                       else vsize[i] = vsize[i] + 1 end
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






