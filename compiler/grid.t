import "compiler.liszt"

local Grid = {}
Grid.__index = Grid
package.loaded["compiler.grid"] = Grid

local L = terralib.require "compiler.lisztlib"

-- There are N x M cells for an NxM grid
local function setupCells(grid)
    local xsize, ysize      = grid:xSize(), grid:ySize()
    local cell_width        = grid:cellWidth()
    local cell_height       = grid:cellHeight()
    local xorigin           = grid:xOrigin()
    local yorigin           = grid:yOrigin()
    local n_bd              = grid:boundaryCells()


    grid.cells:NewFieldMacro('__apply_macro',
        L.NewMacro(function(c,xoff,yoff)
            return liszt `
                L.UNSAFE_ROW( L.id(c) + yoff * xsize + xoff,  grid.cells )
        end))
    
    grid.cells:NewFieldMacro('left', L.NewMacro(function(c)
        return liszt `c(-1, 0)
    end))
    grid.cells:NewFieldMacro('right', L.NewMacro(function(c)
        return liszt `c(1, 0)
    end))
    grid.cells:NewFieldMacro('down', L.NewMacro(function(c)
        return liszt `c(0, -1)
    end))
    grid.cells:NewFieldMacro('up', L.NewMacro(function(c)
        return liszt `c(0, 1)
    end))

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

    -- Set up the boundary!
    grid.cells:NewFieldMacro('is_left_bnd', L.NewMacro(function(c)
        return liszt `c.xid < n_bd
    end))
    grid.cells:NewFieldMacro('is_right_bnd', L.NewMacro(function(c)
        return liszt `c.xid >= xsize - n_bd
    end))
    grid.cells:NewFieldMacro('is_down_bnd', L.NewMacro(function(c)
        return liszt `c.yid < n_bd
    end))
    grid.cells:NewFieldMacro('is_up_bnd', L.NewMacro(function(c)
        return liszt `c.yid >= ysize - n_bd
    end))
    grid.cells:NewFieldMacro('is_bnd', L.NewMacro(function(c)
        return liszt ` c.is_left_bnd or c.is_right_bnd or
                       c.is_up_bnd   or c.is_down_bnd
    end))

    -- Aliases
    grid.cells:NewFieldMacro('in_boundary_region', L.NewMacro(function(c)
        return liszt ` c.is_bnd
    end))
    grid.cells:NewFieldMacro('in_interior', L.NewMacro(function(c)
        return liszt ` not c.is_bnd
    end))
end

-- There are N-1 x M-1 dual cells for an NxM grid
-- By choosing N-1 rather than N+1 we can avoid any special
-- boundary cases for dual cells
local function setupDualCells(grid)
    -- sizes and origins are set for dual cells
    local dxsize, dysize    = grid:xSize() - 1, grid:ySize() - 1
    local cell_width        = grid:cellWidth()
    local cell_height       = grid:cellHeight()
    local dxorigin          = grid:xOrigin() + cell_width/2
    local dyorigin          = grid:yOrigin() + cell_height/2

    local epsilon = 1.0e-5 * math.max(cell_width, cell_height)
    local min_x = dxorigin + epsilon
    local max_x = dxorigin + dxsize * cell_width - epsilon
    local min_y = dyorigin + epsilon
    local max_y = dyorigin + dysize * cell_height - epsilon
    grid.snap_to_grid = L.NewMacro(function(p)
        return liszt quote
            var xy : L.vec2f = p
            if      xy[0] < min_x then xy[0] = L.float(min_x)
            elseif  xy[0] > max_x then xy[0] = L.float(max_x) end
            if      xy[1] < min_y then xy[1] = L.float(min_y)
            elseif  xy[1] > max_y then xy[1] = L.float(max_y) end
        in
            xy
        end
    end)

    grid.dual_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xy = xy_vec -- prevent duplication
            var xidx = L.addr((xy[0] - dxorigin)/cell_width)
            var yidx = L.addr((xy[1] - dyorigin)/cell_height)
        in
            L.UNSAFE_ROW(xidx + yidx * dxsize, grid.dual_cells)
        end
    end)

    grid.dual_cells:NewFieldMacro('xid', L.NewMacro(function(dc)
        return liszt ` L.id(dc) % dxsize
    end))
    grid.dual_cells:NewFieldMacro('yid', L.NewMacro(function(dc)
        return liszt ` L.id(dc) / dxsize
    end))
    grid.dual_cells:NewFieldMacro('center', L.NewMacro(function(dc)
        return liszt `L.vec2f({
            dxorigin +  cell_width * (L.double(dc.xid) + 0.5),
            dyorigin + cell_height * (L.double(dc.yid) + 0.5)
        })
    end))
end

-- There are N+1 x M+1 vertices for an NxM grid
local function setupVertices(grid)
    local vxsize, vysize    = grid:xSize() + 1, grid:ySize() + 1

    grid.vertices:NewFieldMacro('__apply_macro',
        L.NewMacro(function(v, xoff, yoff)
            return liszt `
                L.UNSAFE_ROW( L.id(v) + yoff * vxsize + xoff,  grid.vertices )
        end))
    grid.vertices:NewFieldMacro('left', L.NewMacro(function(v)
        return liszt `v(-1, 0)
    end))
    grid.vertices:NewFieldMacro('right', L.NewMacro(function(v)
        return liszt `v(1, 0)
    end))
    grid.vertices:NewFieldMacro('down', L.NewMacro(function(v)
        return liszt `v(0, -1)
    end))
    grid.vertices:NewFieldMacro('up', L.NewMacro(function(v)
        return liszt `v(0, 1)
    end))

    grid.vertices:NewFieldMacro('xid', L.NewMacro(function (v)
        return liszt ` L.id(v) % vxsize
    end))
    grid.vertices:NewFieldMacro('yid', L.NewMacro(function (v)
        return liszt ` L.id(v) / vxsize
    end))

    -- boundary
    grid.vertices:NewFieldMacro('has_left', L.NewMacro(function(v)
        return liszt `L.id(v) % (vxsize) ~= 0
    end))
    grid.vertices:NewFieldMacro('has_right', L.NewMacro(function(v)
        return liszt `L.id(v) % (vxsize) ~= vxsize-1
    end))
    grid.vertices:NewFieldMacro('has_down', L.NewMacro(function(v)
        return liszt `L.id(v) / (vxsize) ~= 0
    end))
    grid.vertices:NewFieldMacro('has_up', L.NewMacro(function(v)
        return liszt `L.id(v) / (vxsize) ~= vysize-1
    end))
end

-- edges are directed
local function setupEdges(grid)
    -- Indexing Scheme for NxM grid
    -- We offset to 4 regions of the edge indexing space
    -- The regions are:
    --  1. right edges
    --  2. left edges
    --  3. down edges
    --  4. up edges
    local xsize     = grid:xSize()
    local ysize     = grid:ySize()
    local right_off  = 0
    local left_off  = right_off + xsize * (ysize+1)
    local down_off  = left_off  + ysize * (xsize+1)
    local up_off    = down_off  + ysize * (xsize+1)

    grid.edges:NewFieldMacro('is_horizontal', L.NewMacro(function(e)
        return liszt ` L.id(e) < down_off
    end))
    grid.edges:NewFieldMacro('is_vertical', L.NewMacro(function(e)
        return liszt ` L.id(e) >= down_off
    end))
    grid.edges:NewFieldMacro('is_right', L.NewMacro(function(e)
        return liszt ` L.id(e) < left_off
    end))
    grid.edges:NewFieldMacro('is_left', L.NewMacro(function(e)
        return liszt ` e.is_horizontal and L.id(e) >= left_off
    end))
    grid.edges:NewFieldMacro('is_down', L.NewMacro(function(e)
        return liszt ` e.is_vertical and L.id(e) < up_off
    end))
    grid.edges:NewFieldMacro('is_up', L.NewMacro(function(e)
        return liszt ` L.id(e) >= up_off
    end))

    grid.edges:NewFieldMacro('flip', L.NewMacro(function(e)
        return liszt quote
            var f : L.addr= L.id(e)
            if e.is_horizontal then
                if e.is_right then f += left_off else f -= left_off end
            else -- is vertical
                if e.is_down  then f += up_off   else f -= up_off   end
            end
        in
            L.UNSAFE_ROW( f, grid.edges )
        end
    end))
end

local function setupXYedges(grid)
    local cxsize = grid:xSize()
    local cysize = grid:ySize()
    local n_bd   = grid:boundaryCells()
    local ex_xsize = cxsize
    local ex_ysize = cysize - 1
    local ey_xsize = cxsize - 1
    local ey_ysize = cysize

    -- Make these private?
    grid.x_edges:NewFieldMacro('xid', L.NewMacro(function(e)
        return liszt ` L.id(e) % ex_xsize
    end))
    grid.x_edges:NewFieldMacro('yid', L.NewMacro(function(e)
        return liszt ` L.id(e) / ex_xsize
    end))
    grid.y_edges:NewFieldMacro('xid', L.NewMacro(function(e)
        return liszt ` L.id(e) % ey_xsize
    end))
    grid.y_edges:NewFieldMacro('yid', L.NewMacro(function(e)
        return liszt ` L.id(e) / ey_xsize
    end))

    -- edge and cell macros

    local function e_to_cell(e, x, y)
        return liszt `
            L.UNSAFE_ROW( (e.xid + x) + (e.yid + y) * cxsize, grid.cells )
    end

    grid.x_edges:NewFieldMacro('cell_next', L.NewMacro(function(e)
        return e_to_cell(e, 0, 1)
    end))
    grid.x_edges:NewFieldMacro('cell_previous', L.NewMacro(function(e)
        return e_to_cell(e, 0, 0)
    end))
    grid.x_edges:NewFieldMacro('axis', L.NewMacro(function(e)
        return liszt `0
    end))

    grid.y_edges:NewFieldMacro('cell_next', L.NewMacro(function(e)
        return e_to_cell(e, 1, 0)
    end))
    grid.y_edges:NewFieldMacro('cell_previous', L.NewMacro(function(e)
        return e_to_cell(e, 0, 0)
    end))
    grid.y_edges:NewFieldMacro('axis', L.NewMacro(function(e)
        return liszt `1
    end))


    local function cell_to_xe(c, x, y)
        return liszt `
            L.UNSAFE_ROW( (c.xid+x) + (c.yid+y) * ex_xsize, grid.x_edges )
    end
    local function cell_to_ye(c, x, y)
        return liszt `
            L.UNSAFE_ROW( (c.xid+x) + (c.yid+y) * ey_xsize, grid.y_edges )
    end
    grid.cells:NewFieldMacro('edge_up', L.NewMacro(function(c)
        return cell_to_xe(c, 0, 0)
    end))

    grid.cells:NewFieldMacro('edge_down', L.NewMacro(function(c)
        return cell_to_xe(c, 0, -1)
    end))

    grid.cells:NewFieldMacro('edge_right', L.NewMacro(function(c)
        return cell_to_ye(c, 0, 0)
    end))

    grid.cells:NewFieldMacro('edge_left', L.NewMacro(function(c)
        return cell_to_ye(c, -1, 0)
    end))

    grid.x_edges:NewFieldMacro('in_boundary_region', L.NewMacro(function(e)
        return liszt ` e.xid < n_bd or
                       e.xid >= ex_xsize - n_bd or
                       e.yid < n_bd or
                       e.yid >= ex_ysize - n_bd
    end))
    grid.y_edges:NewFieldMacro('in_boundary_region', L.NewMacro(function(e)
        return liszt ` e.xid < n_bd or
                       e.xid >= ey_xsize - n_bd or
                       e.yid < n_bd or
                       e.yid >= ey_ysize - n_bd
    end))
    grid.x_edges:NewFieldMacro('in_interior', L.NewMacro(function(e)
        return liszt `(not e.in_boundary_region)
    end))

    grid.y_edges:NewFieldMacro('in_interior', L.NewMacro(function(e)
        return liszt `(not e.in_boundary_region)
    end))
end

local function setupInterconnects(grid)
    local cxsize, cysize    = grid:xSize(), grid:ySize()
    local vxsize, vysize    = cxsize + 1, cysize + 1

    local function dc_helper (dc, xadd, yadd)
        return liszt ` L.UNSAFE_ROW( (dc.xid + xadd) +
                                     (dc.yid + yadd) * cxsize,  grid.cells )
    end
    grid.dual_cells:NewFieldMacro('downleft', L.NewMacro(function(dc)
        return dc_helper(dc, 0, 0)
    end))
    grid.dual_cells:NewFieldMacro('downright', L.NewMacro(function(dc)
        return dc_helper(dc, 1, 0)
    end))
    grid.dual_cells:NewFieldMacro('upleft', L.NewMacro(function(dc)
        return dc_helper(dc, 0, 1)
    end))
    grid.dual_cells:NewFieldMacro('upright', L.NewMacro(function(dc)
        return dc_helper(dc, 1, 1)
    end))

    grid.dual_cells:NewFieldMacro('vertex', L.NewMacro(function(dc)
        return liszt ` L.UNSAFE_ROW(
            (dc.xid + 1) + (dc.yid + 1) * vxsize,  grid.vertices )
    end))

    -- edge sub-section offsets
    local right_off = 0
    local left_off  = right_off + cxsize * vysize
    local down_off  = left_off  + vxsize * cysize
    local up_off    = down_off  + vxsize * cysize

    local function edge_helper (v, offset, stride, xadd, yadd)
        return liszt ` L.UNSAFE_ROW(
            offset + (v.yid+yadd) * stride + (v.xid+xadd), grid.edges )
    end
    grid.vertices:NewFieldMacro('right_edge', L.NewMacro(function(v)
        return edge_helper(v, right_off, cxsize, 0, 0)
    end))
    grid.vertices:NewFieldMacro('left_edge', L.NewMacro(function(v)
        return edge_helper(v, left_off, cxsize, -1, 0)
    end))
    grid.vertices:NewFieldMacro('down_edge', L.NewMacro(function(v)
        return edge_helper(v, down_off, vxsize, 0, 0)
    end))
    grid.vertices:NewFieldMacro('up_edge', L.NewMacro(function(v)
        return edge_helper(v, up_off, vxsize, 0, -1)
    end))

    -- should set up a way to get the head and tail of edges,
    -- but feeling lazy right now
end


function Grid.New2dUniformGrid(xSize, ySize, pos, w, h, boundary)
    boundary = boundary or 1
    if not xSize or not ySize then
        error('must supply the x and y size of the grid', 2)
    end

    local nCells        = xSize * ySize
    local nDualCells    = (xSize - 1) * (ySize - 1)
    local nVertices     = (xSize + 1) * (ySize + 1)
    local nEdges        = 2 * ( xSize*(ySize+1) + ySize*(xSize+1) )
    local nXEdges       = xSize * (ySize - 1)
    local nYEdges       = (xSize - 1) * ySize
    -- we use two grid systems for the edges

    local grid = setmetatable({
        xdim = xSize,
        ydim = ySize,
        grid_origin = pos,
        grid_width = w,
        grid_height = h,
        cells       = L.NewRelation(nCells, 'cells'),
        dual_cells  = L.NewRelation(nDualCells, 'dual_cells'),
        vertices    = L.NewRelation(nVertices, 'vertices'),
        edges       = L.NewRelation(nEdges, 'edges'),
        -- x_edges are horizontal edges
        x_edges     = L.NewRelation(nXEdges, 'x_edges'),
        -- y_edges are vertical edges
        y_edges     = L.NewRelation(nYEdges, 'y_edges'),
        boundary_cells = boundary,
    }, Grid)

    setupCells(grid)
    setupDualCells(grid)
    setupVertices(grid)
    setupEdges(grid)
    setupXYedges(grid)

    setupInterconnects(grid)

    return grid
end

function Grid:xSize()
    return self.xdim
end

function Grid:ySize()
    return self.ydim
end

function Grid:xOrigin()
    return self.grid_origin[1]
end

function Grid:yOrigin()
    return self.grid_origin[2]
end

function Grid:width()
    return self.grid_width
end

function Grid:height()
    return self.grid_height
end

function Grid:cellWidth()
    return self:width() / (1.0 * self:xSize())
end

function Grid:cellHeight()
    return self:height() / (1.0 * self:ySize())
end

function Grid:boundaryCells()
    return self.boundary_cells
end

