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
            var new_id = id + yoff * xsize + xoff
        in
            L.UNSAFE_ROW(new_id, grid.cells)
        end
        -- TODO: No null checking or plan for that
    end)
    

    grid.cells:NewFieldMacro('left', L.NewMacro(function(c)
        return liszt quote
            var raw_addr = L.id(c)
            -- TODO: The following assert should check the entire
            -- left side of the grid, not just the top left
--            assert(raw_addr > 0)
        in
            L.UNSAFE_ROW( raw_addr - 1, grid.cells )
        end
    end))

    grid.cells:NewFieldMacro('right', L.NewMacro(function(c)
        return liszt quote
            var raw_addr = L.id(c)
            -- TODO: The following assert should check the entire
            -- left side of the grid, not just the top left
--            assert(raw_addr > 0)
        in
            L.UNSAFE_ROW( raw_addr + 1, grid.cells )
        end
    end))

    grid.cells:NewFieldMacro('top', L.NewMacro(function(c)
        return liszt quote
            var raw_addr = L.id(c)
            -- TODO: The following assert should check the entire
            -- left side of the grid, not just the top left
--            assert(raw_addr > 0)
        in
            L.UNSAFE_ROW( raw_addr - xsize, grid.cells )
        end
    end))

    grid.cells:NewFieldMacro('bot', L.NewMacro(function(c)
        return liszt quote
            var raw_addr = L.id(c)
            -- TODO: The following assert should check the entire
            -- left side of the grid, not just the top left
--            assert(raw_addr > 0)
        in
            L.UNSAFE_ROW( raw_addr + xsize, grid.cells )
        end
    end))

    grid.cells:NewFieldMacro('center', L.NewMacro(function(c)
        return liszt quote
            var raw_addr    = L.id(c)
            var y_id        = raw_addr / L.addr(xsize)
            var x_id        = raw_addr - y_id * L.addr(ysize)
        in
            L.vec2f({
                xorigin + cell_width  * (L.float(x_id) + 0.5),
                yorigin + cell_height * (L.float(y_id) + 0.5)
            })
        end
    end))


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

    grid.dual_cells:NewFieldMacro('topleft', L.NewMacro(function(dc)
        return liszt quote
            var raw_addr = L.id(dc)
            var y_id     = raw_addr / L.addr(xsize-1)
            var x_id     = raw_addr - y_id * L.addr(xsize-1)
        in
            L.UNSAFE_ROW( x_id + y_id * xsize,  grid.cells )
        end
    end))

    grid.dual_cells:NewFieldMacro('topright', L.NewMacro(function(dc)
        return liszt quote
            var raw_addr = L.id(dc)
            var y_id     = raw_addr / L.addr(xsize-1)
            var x_id     = raw_addr - y_id * L.addr(xsize-1)
        in
            L.UNSAFE_ROW( (x_id + 1) + y_id * xsize,  grid.cells )
        end
    end))

    grid.dual_cells:NewFieldMacro('bottomleft', L.NewMacro(function(dc)
        return liszt quote
            var raw_addr = L.id(dc)
            var y_id     = raw_addr / L.addr(xsize-1)
            var x_id     = raw_addr - y_id * L.addr(xsize-1)
            var new_id   = x_id + (y_id+1) * xsize
        in
            L.UNSAFE_ROW( (x_id + 1) + y_id * xsize,  grid.cells )
        end
    end))

    grid.dual_cells:NewFieldMacro('bottomright', L.NewMacro(function(dc)
        return liszt quote
            var raw_addr = L.id(dc)
            var y_id     = raw_addr / L.addr(xsize-1)
            var x_id     = raw_addr - y_id * L.addr(xsize-1)
            var new_id   = x_id + (y_id+1) * xsize
        in
            L.UNSAFE_ROW( (x_id + 1) + (y_id + 1) * xsize,  grid.cells )
        end
    end))

--    grid.cells_locate = L.NewMacro(function(xy_pos_vec)
--        -- compute any constants???
--        local xsize = grid:xSize()
--
--        return liszt quote
--            var id = L.id(c)
--            -- Should actually do conversion from coordinates to integers...
--            var xoff = xy_pos_vec[0]
--            var yoff = xy_pos_vec[1]
--
--            var new_id = id + yoff * xsize + xoff
--        in
--            L.UNSAFE_ROW(new_id, grid.cells)
--        end
--    end)
end

--local initPrivateIndices = liszt_kernel(c: grid.cells)
--    c.private.index = c.getPrivateIndex()
--end

function Grid.New2dUniformGrid(xSize, ySize, pos, w, h)
    if not xSize or not ySize then
        error('must supply the x and y size of the grid', 2)
    end

    local nCells = xSize * ySize
    local nDualCells = (xSize - 1) * (ySize - 1)

    local grid = setmetatable({
        xdim = xSize,
        ydim = ySize,
        grid_origin = pos,
        grid_width = w,
        grid_height = h,
        cells       = L.NewRelation(nCells, 'cells'),
        dual_cells  = L.NewRelation(nDualCells, 'dual_cells'),
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



