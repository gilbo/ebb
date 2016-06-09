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
cells:partition { blocking = {2,2} }

local mass = ewrap.NewField {
  name = 'mass',
  rel  = cells,
  type = float,
}

-- load field
--remoteLoadFieldConstant('cells', 'mass', {{1}})

-- load field again
--remoteLoadFieldConstant('cells', 'mass', {{3.2}})

print("*** Completed control pogram.")
