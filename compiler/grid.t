import "compiler.liszt"

local Grid = {}
Grid.__index = Grid
package.loaded["compiler.grid"] = Grid

local L = terralib.require "compiler.lisztlib"

local function installMacros(grid)

    local xsize, ysize      = grid:xSize(), grid:ySize()
    local cell_width        = grid:cellWidth()
    local cell_height       = grid:cellHeight()
    local xorigin           = grid:xOrigin()
    local yorigin           = grid:yOrigin()
    local boundary_cells    = grid:boundaryCells()

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
    grid.offset = L.NewMacro(function(c, xoff, yoff)
        return liszt quote
            var id = L.id(c)
        in
            L.UNSAFE_ROW( id + yoff * xsize + xoff,  grid.cells )
        end
        -- TODO: No null checking or plan for that
    end)
    
    grid.cells:NewFieldMacro('left', L.NewMacro(function(c)
        return liszt `grid.offset(c, -1, 0)
    end))
    grid.cells:NewFieldMacro('right', L.NewMacro(function(c)
        return liszt `grid.offset(c, 1, 0)
    end))
    grid.cells:NewFieldMacro('down', L.NewMacro(function(c)
        return liszt `grid.offset(c, 0, -1)
    end))
    grid.cells:NewFieldMacro('up', L.NewMacro(function(c)
        return liszt `grid.offset(c, 0, 1)
    end))

    -- Should this be hidden?
    grid.cells:NewFieldMacro('xy_ids', L.NewMacro(function(c)
        return liszt quote
            var raw_addr    = L.id(c)
            var y_id        = raw_addr / L.addr(xsize)
            var x_id        = raw_addr - y_id * L.addr(xsize)
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

    local epsilon = 1.0e-5 * math.max(cell_width, cell_height)
    local min_x = xorigin + (0.5 * cell_width) + epsilon
    local max_x = xorigin + xsize * cell_width + 0.5 * cell_width - epsilon
    local min_y = yorigin + (0.5 * cell_height) + epsilon
    local max_y = yorigin + ysize * cell_height + 0.5 * cell_height - epsilon
    grid.snap_to_grid = L.NewMacro(function(p)
        return liszt quote
            var xy : L.vec2f = p
            if      xy[0] < min_x then xy[0] = L.float(min_x)
            elseif  xy[0] > max_x then xy[0] = L.float(max_x) end
            if      xy[1] < min_y then xy[1] = L.float(min_y)
            elseif  xy[1] > max_y then xy[1] = L.float(max_y) end
        in
            L.vec2f(xy)
        end
    end)


    grid.dual_locate = L.NewMacro(function(xy_vec)
        return liszt quote
            var xidx = L.int((xy_vec[0] - xorigin)/cell_width - 0.5)
            var yidx = L.int((xy_vec[1] - yorigin)/cell_height - 0.5)

            -- xsize-1 for dual grid...
            var dual_id : L.addr = xidx + yidx * (xsize-1)
        in
            L.UNSAFE_ROW(dual_id, grid.dual_cells)
        end
    end)


    grid.dual_cells:NewFieldMacro('center', L.NewMacro(function(dc)
        return liszt quote
            var raw_addr = L.id(dc)
            var y_id     = raw_addr / L.addr(xsize-1)
            var x_id     = raw_addr - y_id * L.addr(xsize-1)
            var xy       = L.vec2d({ x_id + 1.0, y_id + 1.0 })
        in
            L.vec2f({ xorigin +  cell_width * xy[0],
                      yorigin + cell_height * xy[1] })
        end
    end))

    local function dc_helper (dc, xadd, yadd)
        return liszt quote
            var raw_addr = L.id(dc)
            var y_id     = raw_addr / L.addr(xsize-1)
            var x_id     = raw_addr - y_id * L.addr(xsize-1)
        in
            L.UNSAFE_ROW( (x_id + xadd) + (y_id + yadd) * xsize,  grid.cells )
        end
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

    -- Set up the boundary!
    grid.cells:NewFieldMacro('is_left_bnd', L.NewMacro(function(c)
        return liszt `(c.xy_ids[0] == 0)
    end))
    grid.cells:NewFieldMacro('is_right_bnd', L.NewMacro(function(c)
        return liszt `(c.xy_ids[0] == xsize-1)
    end))
    grid.cells:NewFieldMacro('is_down_bnd', L.NewMacro(function(c)
        return liszt `(c.xy_ids[1] == 0)
    end))
    grid.cells:NewFieldMacro('is_up_bnd', L.NewMacro(function(c)
        return liszt `(c.xy_ids[1] == ysize-1)
    end))
    grid.cells:NewFieldMacro('is_bnd', L.NewMacro(function(c)
        return liszt ` c.is_left_bnd or c.is_right_bnd or
                       c.is_up_bnd   or c.is_down_bnd
    end))

    -- edge and cell macros

    local function xe_to_cell(e, dir)
        return liszt quote
            var raw_addr = L.id(e)
            var y_id = raw_addr / L.addr(xsize)
            var x_id = raw_addr - y_id * L.addr(xsize)
        in
            L.UNSAFE_ROW( x_id + (y_id + dir) * xsize, grid.cells)
        end
    end

    grid.x_edges:NewFieldMacro('cell_next', L.NewMacro(function(e)
        return xe_to_cell(e, 1)
    end))

    grid.x_edges:NewFieldMacro('cell_previous', L.NewMacro(function(e)
        return xe_to_cell(e, 0)
    end))

    grid.x_edges:NewFieldMacro('axis', L.NewMacro(function(e)
        return liszt `0
    end))

    local function ye_to_cell(e, dir)
        return liszt quote
            var raw_addr = L.id(e)
            var y_id = raw_addr / L.addr(xsize-1)
            var x_id = raw_addr - y_id * L.addr(xsize-1)
        in
            L.UNSAFE_ROW( (x_id + dir) + y_id * xsize, grid.cells)
        end
    end

    grid.y_edges:NewFieldMacro('cell_next', L.NewMacro(function(e)
        return ye_to_cell(e, 1)
    end))

    grid.y_edges:NewFieldMacro('cell_previous', L.NewMacro(function(e)
        return ye_to_cell(e, 0)
    end))

    grid.y_edges:NewFieldMacro('axis', L.NewMacro(function(e)
        return liszt `1
    end))


    local function cell_to_xe(c, dir)
        return liszt quote
            var raw_addr = L.id(c)
            var y_id = raw_addr / L.addr(xsize)
            var x_id = raw_addr - y_id * L.addr(xsize)
        in
            L.UNSAFE_ROW( x_id + (y_id-1 + dir) * xsize, grid.x_edges )
        end
    end

    local function cell_to_ye(c, dir)
        return liszt quote
            var raw_addr = L.id(c)
            var y_id = raw_addr / L.addr(xsize)
            var x_id = raw_addr - y_id * L.addr(xsize)
        in
            L.UNSAFE_ROW( (x_id-1 + dir) + y_id * xsize, grid.y_edges )
        end
    end

    grid.cells:NewFieldMacro('edge_up', L.NewMacro(function(c)
        return cell_to_xe(c, 1)
    end))

    grid.cells:NewFieldMacro('edge_down', L.NewMacro(function(c)
        return cell_to_xe(c, 0)
    end))

    grid.cells:NewFieldMacro('edge_right', L.NewMacro(function(c)
        return cell_to_ye(c, 1)
    end))

    grid.cells:NewFieldMacro('edge_left', L.NewMacro(function(c)
        return cell_to_ye(c, 0)
    end))


    -- boundary macros for higher stencils

    grid.cells:NewFieldMacro('in_boundary_region', L.NewMacro(function(c)
        return liszt `(c.xy_ids[0] < boundary_cells or
                       c.xy_ids[0] > xsize - 1 - boundary_cells or
                       c.xy_ids[1] < boundary_cells or
                       c.xy_ids[1] > ysize - 1 - boundary_cells)
    end))

    grid.cells:NewFieldMacro('in_interior', L.NewMacro(function(c)
        return liszt `(not c.in_boundary_region)
    end))

    grid.x_edges:NewFieldMacro('xy_ids', L.NewMacro(function(e)
        return liszt quote
            var raw_addr = L.id(e)
            var y_id     = raw_addr / L.addr(xsize)
            var x_id     = raw_addr - y_id * L.addr(xsize)
        in
            { x_id, y_id }
        end
    end))

    grid.y_edges:NewFieldMacro('xy_ids', L.NewMacro(function(e)
        return liszt quote
            var raw_addr = L.id(e)
            var y_id     = raw_addr / L.addr(xsize-1)
            var x_id     = raw_addr - y_id * L.addr(xsize-1)
        in
            { x_id, y_id }
        end
    end))

    grid.x_edges:NewFieldMacro('in_boundary_region', L.NewMacro(function(e)
        return liszt `(e.xy_ids[0] < boundary_cells or
                       e.xy_ids[0] > xsize - 1 - boundary_cells or
                       e.xy_ids[1] < boundary_cells or
                       e.xy_ids[1] > ysize - 2 - boundary_cells)
    end))

    grid.y_edges:NewFieldMacro('in_boundary_region', L.NewMacro(function(e)
        return liszt `(e.xy_ids[0] < boundary_cells or
                       e.xy_ids[0] > xsize - 2 - boundary_cells or
                       e.xy_ids[1] < boundary_cells or
                       e.xy_ids[1] > ysize - 1 - boundary_cells)
    end))

    grid.x_edges:NewFieldMacro('in_interior', L.NewMacro(function(e)
        return liszt `(not e.in_boundary_region)
    end))

    grid.y_edges:NewFieldMacro('in_interior', L.NewMacro(function(e)
        return liszt `(not e.in_boundary_region)
    end))

end

--local initPrivateIndices = liszt_kernel(c: grid.cells)
--    c.private.index = c.getPrivateIndex()
--end

function Grid.New2dUniformGrid(xSize, ySize, pos, w, h, boundary)
    boundary = boundary or 0
    if not xSize or not ySize then
        error('must supply the x and y size of the grid', 2)
    end

    local nCells = xSize * ySize
    local nDualCells = (xSize - 1) * (ySize - 1)
    local nXEdges = xSize * (ySize - 1)
    local nYEdges = (xSize - 1) * ySize

    local grid = setmetatable({
        xdim = xSize,
        ydim = ySize,
        grid_origin = pos,
        grid_width = w,
        grid_height = h,
        cells       = L.NewRelation(nCells, 'cells'),
        dual_cells  = L.NewRelation(nDualCells, 'dual_cells'),
        x_edges     = L.NewRelation(nXEdges, 'x_edges'),
        y_edges     = L.NewRelation(nYEdges, 'y_edges'),
        boundary_cells = boundary,
    }, Grid)

    installMacros(grid)

--    grid.cells:NewField('private.index', L.vector(L.int, {0,0}))

    -- TODO: init all indices
--    initPrivateIndices(grid.cells)

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
