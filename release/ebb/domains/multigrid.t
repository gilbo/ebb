import "ebb.liszt"

local MultiGrid = {}
package.loaded["ebb.domains.multigrid"] = MultiGrid

local Multi2d = {}
Multi2d.__index = Multi2d



local Grid = require 'ebb.domains.grid'


local function is_power_of_two(n)
  while n > 1 do
    if n%2 ~= 0 then return false end
    n = n/2
  end
  return n == 1
end



function build_next_level(cellrel)
  local Ndown = cellrel:Dims()[1]
  local Nup   = Ndown / 2
  local grid = Grid.NewGrid2d {
    size    = {Nup, Nup},
    origin  = {0,0},
    width   = {Nup, Nup},
    periodic_boundary = {true,true},
  }

  -- link the cells of down to up
  cellrel:NewFieldFunction('up_cell', liszt (c)
    return L.Affine(grid.cells, {{ 0.5,   0, 0},
                                 {   0, 0.5, 0}}, c)
  end)
  -- link the cells of up to down
  grid.cells:NewFieldFunction('down_cell', liszt (c)
    return L.Affine(cellrel, {{ 2, 0, 0},
                              { 0, 2, 0}}, c)
  end)

  return grid
end

function MultiGrid.NewMultiGrid2d(params)
local calling_convention = [[
NewMultiGrid2d should be called with named parameters:
MultiGrid.NewMultiGrid2d {
  base_rel        = relation,       -- a Grid mode relation to
                                    -- construct the multigrid on
  top_resolution  = #,              -- # of cells in X and Y at the top
                                    -- level of the multigrid hierarchy
]]
  local function check_params(params)
    if type(params) ~= 'table' then return false end
    if not L.is_relation(params.base_rel) then return false end
    return true
  end
  if not check_params(params) then error(calling_convention, 2) end

  local dims = params.base_rel:Dims()
  if #dims ~= 2 then
    error('NewMultiGrid2d expects a 2d grid base relation', 2)
  end
  if dims[1] ~= dims[2] then
    error('NewMultiGrid2d expects a square grid right now', 2)
  end
  if not is_power_of_two(dims[1]) then
    error('NewMultiGrid2d currently only supports grids with '..
          '2^n cells in the X and Y dimensions (for simplicity)', 2)
  end
  local periodicity = params.base_rel:Periodicity()
  if not periodicity[1] or not periodicity[2] then
    error('NewMultiGrid2d currently only supports periodic relations', 2)
  end

  local base_rel = params.base_rel
  local top_res  = params.top_resolution or 8

  local levels = {}

  local rel = base_rel
  local grid = build_next_level(rel)
  table.insert(levels, grid)
  while grid:xSize() > top_res do
    rel = grid.cells
    grid = build_next_level(rel)
    table.insert(levels, grid)
  end

  local mg = setmetatable({
    _levels     = levels
  }, Multi2d)
  return mg
end


function Multi2d:nLevels()              return #self._levels       end
function Multi2d:level(k)               return self._levels[k]     end

function Multi2d:levelIter()
  local i=0

  return function()
    i = i + 1
    if i > #self._levels then return nil end
    return self._levels[i]
  end
end



















