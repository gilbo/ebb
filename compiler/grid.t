import "compiler.liszt"

local Grid = {}
Grid.__index = Grid
package.loaded["compiler.grid"] = Grid

local L = terralib.require "compiler.lisztlib"



local function installMacros(grid)
    --grid.cells:NewFieldMacro('offset', L.NewMacro(function(c, xoff, yoff)
    --    -- TODO should check that xoff/yoff are number literals here...
    --    local xsize = grid:xSize()
    --    return liszt `begin
    --        var id = L.id(c)
    --        var new_id = id + yoff * xsize + xoff
    --        return L.UNSAFE_ROW(new_id, grid.cells)
    --    end
    --    -- TODO: somehow return null? when there is no cell?
    --end))
    
    grid.cells:NewFieldMacro('left', L.NewMacro(function(c)
        return liszt `L.UNSAFE_ROW( L.id(c)-1, grid.cells )
    end))
end

function Grid.New2dUniformGrid(xSize, ySize)
    if not xSize or not ySize then
        error('must supply the x and y size of the grid', 2)
    end

    local nCells = xSize * ySize

    local grid = setmetatable({
        xdim = xSize,
        ydim = ySize,
        cells = L.NewRelation(nCells, 'cells'),
    }, Grid)

    installMacros(grid)

    return grid
end

function Grid:xSize()
    return self.xdim
end

function Grid:ySize()
    return self.ydim
end


--[[ NOTE MACRO PROBLEMS: 
    We cannot create a complicated expression, which we need to implement
    offset correctly...
]]



