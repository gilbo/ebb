-- Test partitioning on 2d grid.

import "ebb"
local L = require "ebblib"

print("***************************************************")
print("**  This is an Ebb application intended to test  **")
print("**  partitioning over 2d grid relations          **")
print("***************************************************")

-- includes
local Grid  = require 'ebb.domains.grid'

-- grid
local Nx = 2
local Ny = 2
local grid = Grid.NewGrid2d {
    size   = {Nx, Ny},
    origin = {0, 0},
    width  = {4, 4},
    periodic_boundary = {true, true},
}
local C = grid.cells
local V = grid.vertices

-------------------------------------------------------------------------------
--  Initialization                                                           --
-------------------------------------------------------------------------------

-- field declarations
V:NewField('value', L.int):Load(0)

-------------------------------------------------------------------------------
--  Reduce to field                                                          --
-------------------------------------------------------------------------------

local ebb ReduceToField(c)
  c.vertex(-1,-1).value += 1
  c.vertex(-1, 1).value += 1
  c.vertex( 1,-1).value += 1
  c.vertex( 1, 1).value += 1
end

-------------------------------------------------------------------------------
--  Invoke reduction                                                         --
-------------------------------------------------------------------------------

V.value:Print()
C:foreach(ReduceToField)
V.value:Print()
