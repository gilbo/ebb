#!/usr/bin/env terra

local gas       = require 'gasnet'
local gaswrap   = require 'gaswrap'
local ewrap     = require 'ebb.src.ewrap'

local C = terralib.includecstring [[
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>
]]


-------------------------------------------------------------------------------
-- Control program
-------------------------------------------------------------------------------

local cells = ewrap.NewGridRelation {
  name = 'cells',
  dims = {8,8}
}
cells:partition_across_nodes { blocking = {2,2} }

local mass = ewrap.NewField {
  name = 'mass',
  rel  = cells,
  type = float,
}

print("*** Completed control pogram.")
