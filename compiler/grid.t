import "compiler.liszt"

local Grid = {}

package.loaded["compiler.grid"] = Grid

local L = terralib.require "compiler.lisztlib"

Grid.GridClass = {}
Grid.GridClass.__index = Grid.GridClass

-- mjlgg: Useful table printer
function tprint (tbl, indent)
    if not indent then indent = 0 end
    for k, v in pairs(tbl) do
        local formatting = string.rep("  ", indent) .. k .. ": "
        if type(v) == "table" then
            print(formatting)
            tprint(v, indent+1)
        else
            print(formatting .. tostring(v))
        end
    end
end

function Grid.GridClass:initUniformGrid(dim, value, globals)
    assert(dim)
    assert(#dim == 2 or #dim == 3)

    if value == nil then
        value = 0
    end

    local table = {}
    local grid = setmetatable(table, Grid.GridClass)

    for i = 1, dim[1] do
        table[i] = {}
        for j = 1, dim[2] do
            if #dim == 3 then
                for k = 1, dim[3] do
                    table[i][j][k] = value
                end
            else
                table[i][j] = value
            end
        end
    end
    
    grid['dim'] = dim
    grid['globals'] = globals
    
    return grid
end

function Grid.GridClass:get(index)
    if #index == 2 then
        assert(index[1] <= self['dim'][1])
        assert(index[2] <= self['dim'][2])

        return self[index[1]][index[2]]
    else
        assert(#index == 3)
        assert(#self['dim'] == 3)
        assert(index[1] <= self['dim'][1])
        assert(index[2] <= self['dim'][2])
        assert(index[3] <= self['dim'][3])

        return self[index[1]][index[2]][index[3]]
    end
end

function Grid.GridClass:set(index, value)
    if #index == 2 then
        assert(#self['dim'] == 2)
        assert(index[1] <= self['dim'][1])
        assert(index[2] <= self['dim'][2])

        self[index[1]][index[2]] = value
    else
        assert(#index == 3)
        assert(#self['dim'] == 3)
        assert(index[1] <= self['dim'][1])
        assert(index[2] <= self['dim'][2])
        assert(index[3] <= self['dim'][3])

        self[index[1]][index[2]][index[3]] = value
    end
end

function Grid.GridClass:print()
    tprint(self)
end

