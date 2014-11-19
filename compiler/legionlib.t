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
--[[ Legion typedefs                                                       ]]--
-------------------------------------------------------------------------------

local fid_t = Lc.legion_field_id_t

-------------------------------------------------------------------------------
--[[ LogicalRegion methods                                                 ]]--
-------------------------------------------------------------------------------

local struct LogicalRegion {
  is : Lc.legion_index_space_t;
  isa : Lc.legion_index_allocator_t;
  fs : Lc.legion_field_space_t;
  fsa : Lc.legion_field_allocator_t;
  handle : Lc.legion_logical_region_t;
  rows_max : Tc.size_t;
  rows_alloc : Tc.size_t;
  fields_alloc : fid_t;
}
L.LogicalRegion = LogicalRegion

terra L.NewLogicalRegion(size : Tc.size_t)
  var l : LogicalRegion
  l.is = Lc.legion_index_space_create(@runtime, @ctx, size)
  l.isa = Lc.legion_index_allocator_create(@runtime, @ctx, l.is)
  l.fs = Lc.legion_field_space_create(@runtime, @ctx)
  l.fsa = Lc.legion_field_allocator_create(@runtime, @ctx, l.fs)
  l.handle = Lc.legion_logical_region_create(@runtime, @ctx, l.is, l.fs)
  l.rows_max = size
  l.rows_alloc = 0
  return l
end

terra LogicalRegion:AllocateRows(size : Tc.size_t)
  if self.rows_alloc + size <= self.rows_max then
    Lc.legion_index_allocator_alloc(self.isa, size)
    self.rows_alloc = self.rows_alloc + size
    return true
  else
    return false
  end
end

terra LogicalRegion:AllocateField(field_size : Tc.size_t)
  var field_id : fid_t = self.fields_alloc
  self.fields_alloc = self.fields_alloc + 1
  return Lc.legion_field_allocator_allocate_field(self.fsa,
                                                 field_size, field_id)
end
