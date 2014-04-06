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

-- UNFORTUNATELY OFFSET WON'T WORK CURRENTLY
--    grid.cells:NewFieldMacro('offset', L.NewMacro(function(c, xoff, yoff)
--        -- TODO should check that xoff/yoff are number literals here...
--        --local xsize = grid:xSize()
--        return liszt quote
--            var id = L.id(c)
--            var new_id = id + yoff * grid.xdim + xoff
--        in
--            L.UNSAFE_ROW(new_id, grid.cells)
--        end
--        -- TODO: somehow return null? when there is no cell?
--    end))

-- Workaround for now
    grid.cell_offset = L.NewMacro(function(c, xoff, yoff)
        return liszt `
            L.UNSAFE_ROW( L.id(c) + yoff * xsize + xoff,  grid.cells )
        -- TODO: No null checking or plan for that
    end)
    
    grid.cells:NewFieldMacro('left', L.NewMacro(function(c)
        return liszt `grid.cell_offset(c, -1, 0)
    end))
    grid.cells:NewFieldMacro('right', L.NewMacro(function(c)
        return liszt `grid.cell_offset(c, 1, 0)
    end))
    grid.cells:NewFieldMacro('down', L.NewMacro(function(c)
        return liszt `grid.cell_offset(c, 0, -1)
    end))
    grid.cells:NewFieldMacro('up', L.NewMacro(function(c)
        return liszt `grid.cell_offset(c, 0, 1)
    end))

    -- Should this be hidden?
    grid.cells:NewFieldMacro('xy_ids', L.NewMacro(function(c)
        return liszt quote
            var y_id = L.id(c) / L.addr(xsize)
            var x_id = L.id(c) % L.addr(xsize)
        in
            { x_id, y_id }
        end
    end))

    grid.cells:NewFieldMacro('center', L.NewMacro(function(c)
        return liszt quote
            var xy = L.vec2d(c.xy_ids) + {0.5, 0.5}
        in
            L.vec2f({ xorigin, yorigin } +
                    { cell_width * xy[0], cell_height * xy[1] })
        end
    end))

    -- Set up the boundary!
    grid.cells:NewFieldMacro('is_left_bnd', L.NewMacro(function(c)
        return liszt `L.id(c) % xsize == 0
    end))
    grid.cells:NewFieldMacro('is_right_bnd', L.NewMacro(function(c)
        return liszt `L.id(c) % xsize == xsize-1
    end))
    grid.cells:NewFieldMacro('is_down_bnd', L.NewMacro(function(c)
        return liszt `L.id(c) / xsize == 0
    end))
    grid.cells:NewFieldMacro('is_up_bnd', L.NewMacro(function(c)
        return liszt `L.id(c) / xsize == ysize-1
    end))
    grid.cells:NewFieldMacro('is_bnd', L.NewMacro(function(c)
        return liszt ` c.is_left_bnd or c.is_right_bnd or
                       c.is_up_bnd   or c.is_down_bnd
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

    grid.dual_cells:NewFieldMacro('xy_ids', L.NewMacro(function (dc)
        return liszt quote
            var y_id = L.id(dc) / L.addr(dxsize)
            var x_id = L.id(dc) % L.addr(dxsize)
        in
            { x_id, y_id }
        end
    end))
    grid.dual_cells:NewFieldMacro('center', L.NewMacro(function(dc)
        return liszt `L.vec2f({
            dxorigin +  cell_width * (L.double(dc.xy_ids[0]) + 0.5),
            dyorigin + cell_height * (L.double(dc.xy_ids[1]) + 0.5)
        })
    end))
end

-- There are N+1 x M+1 vertices for an NxM grid
local function setupVertices(grid)
    local vxsize, vysize    = grid:xSize() + 1, grid:ySize() + 1

    grid.vertex_offset = L.NewMacro(function(v, xoff, yoff)
        return liszt `
            L.UNSAFE_ROW( L.id(v) + yoff * vxsize + xoff,  grid.vertices )
    end)
    grid.vertices:NewFieldMacro('left', L.NewMacro(function(v)
        return liszt `grid.vertex_offset(v, -1, 0)
    end))
    grid.vertices:NewFieldMacro('right', L.NewMacro(function(v)
        return liszt `grid.vertex_offset(v, 1, 0)
    end))
    grid.vertices:NewFieldMacro('down', L.NewMacro(function(v)
        return liszt `grid.vertex_offset(v, 0, -1)
    end))
    grid.vertices:NewFieldMacro('up', L.NewMacro(function(v)
        return liszt `grid.vertex_offset(v, 0, 1)
    end))

    grid.vertices:NewFieldMacro('xy_ids', L.NewMacro(function (v)
        return liszt quote
            var y_id        = L.id(v) / L.addr(vxsize)
            var x_id        = L.id(v) % L.addr(vxsize)
        in
            { x_id, y_id }
        end
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

local function setupInterconnects(grid)
    local cxsize, cysize    = grid:xSize(), grid:ySize()
    local vxsize, vysize    = cxsize + 1, cysize + 1

    local function dc_helper (dc, xadd, yadd)
        return liszt `
            L.UNSAFE_ROW( (dc.xy_ids[0] + xadd) +
                          (dc.xy_ids[1] + yadd) * cxsize,  grid.cells )
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
            (dc.xy_ids[0] + 1) + (dc.xy_ids[1] + 1) * cxsize,  grid.vertices )
    end))

    -- edge sub-section offsets
    local right_off = 0
    local left_off  = right_off + cxsize * vysize
    local down_off  = left_off  + vxsize * cysize
    local up_off    = down_off  + vxsize * cysize

    local function edge_helper (v, offset, stride, xadd, yadd)
        return liszt quote
            var xy           = v.xy_ids
            var eid : L.addr = offset + (xy[1]+yadd) * stride + (xy[0]+xadd)
        in
            L.UNSAFE_ROW( eid, grid.edges )
        end
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


function Grid.New2dUniformGrid(xSize, ySize, pos, w, h)
    if not xSize or not ySize then
        error('must supply the x and y size of the grid', 2)
    end

    local nCells        = xSize * ySize
    local nDualCells    = (xSize - 1) * (ySize - 1)
    local nVertices     = (xSize + 1) * (ySize + 1)
    local nEdges        = 2 * ( xSize*(ySize+1) + ySize*(xSize+1) )
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
        edges       = L.NewRelation(nEdges, 'edges')
    }, Grid)

    setupCells(grid)
    setupDualCells(grid)
    setupVertices(grid)
    setupEdges(grid)

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



