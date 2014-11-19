local L = {}
package.loaded["compiler.legionlib"] = L

local C = terralib.require "compiler.c"

local Tc = terralib.require "compiler.typedefs"
terralib.require "legionlib-terra"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])

-------------------------------------------------------------------------------
--[[ Legion environment                                                    ]]--
-------------------------------------------------------------------------------

local legion_env = rawget(_G, '_legion_env')
local ctx = legion_env.ctx
local runtime = legion_env.runtime

-------------------------------------------------------------------------------
--[[ LogicalRegion methods                                                 ]]--
-------------------------------------------------------------------------------

local struct LogicalRegion {
  index_space : Lc.legion_index_space_t;
  field_space : Lc.legion_field_space_t;
  handle : Lc.legion_logical_region_t;
}
L.LogicalRegion = LogicalRegion

terra L.NewLogicalRegion(size : Tc.size_t)
  var l : LogicalRegion
  l.index_space = Lc.legion_index_space_create(@runtime, @ctx, size)
  l.field_space = Lc.legion_field_space_create(@runtime, @ctx)
  l.handle = Lc.legion_logical_region_create(@runtime, @ctx,
                                             l.index_space, l.field_space)
  return l
end
