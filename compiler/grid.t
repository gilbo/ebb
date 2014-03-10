import "compiler.liszt"

local Grid = {}
Grid.__index = Grid
package.loaded["compiler.grid"] = Grid

local L = terralib.require "compiler.lisztlib"

local function installMacros(grid)
    grid.cells:NewFieldMacro('offset', L.NewMacro(function(c, xoff, yoff)
        -- TODO should check that xoff/yoff are number literals here...
        --local xsize = grid:xSize()
        return liszt quote
            var id = L.id(c)
            var new_id = id + yoff * grid.xdim + xoff
        in
            L.UNSAFE_ROW(new_id, grid.cells)
        end
        -- TODO: somehow return null? when there is no cell?
    end))
    
    grid.cells:NewFieldMacro('left', L.NewMacro(function(c)
        return liszt quote
            var raw_addr = L.id(c)
            -- TODO: The following assert should check the entire
            -- left side of the grid, not just the top left
--            assert(raw_addr > 0)
            --raw_addr -= 1
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
            --raw_addr += 1
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
            --raw_addr -= grid.xdim
        in
            L.UNSAFE_ROW( raw_addr - grid.xdim, grid.cells )
        end
    end))

    grid.cells:NewFieldMacro('bot', L.NewMacro(function(c)
        return liszt quote
            var raw_addr = L.id(c)
            -- TODO: The following assert should check the entire
            -- left side of the grid, not just the top left
--            assert(raw_addr > 0)
            --raw_addr += grid.xdim
        in
            L.UNSAFE_ROW( raw_addr + grid.xdim, grid.cells )
        end
    end))

    grid.cells:NewFieldMacro('locate', L.NewMacro(function(c, xoff, yoff)
        -- TODO should check that xoff/yoff are number literals here...
        --local xsize = grid:xSize()
        return liszt quote
            var id = L.id(c)
            var new_id = id + yoff * grid.xdim + xoff
        in
            L.UNSAFE_ROW(new_id, grid.cells)
        end
        -- TODO: somehow return null? when there is no cell?
    end))
end

--local initPrivateIndices = liszt_kernel(c: grid.cells)
--    c.private.index = c.getPrivateIndex()
--end

function Grid.New2dUniformGrid(xSize, ySize, pos, w, h)
    if not xSize or not ySize then
        error('must supply the x and y size of the grid', 2)
    end

    local nCells = xSize * ySize

    local grid = setmetatable({
        xdim = xSize,
        ydim = ySize,
        position = pos,
        width = w,
        height = h,
        cells = L.NewRelation(nCells, 'cells'),
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

function Grid:position()
    return self.position
end

function Grid:width()
    return self.width
end

function Grid:height()
    return self.height
end

--[[ NOTE MACRO PROBLEMS: 
    We cannot create a complicated expression, which we need to implement
    offset correctly...
]]

