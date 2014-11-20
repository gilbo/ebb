local L = {}
package.loaded["compiler.legionlib"] = L

local C = terralib.require "compiler.c"

local Tc = terralib.require "compiler.typedefs"
terralib.require "legionlib-terra"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])


-------------------------------------------------------------------------------
--[[                          Legion environment                           ]]--
-------------------------------------------------------------------------------

local legion_env = rawget(_G, '_legion_env')
local ctx = legion_env.ctx
local runtime = legion_env.runtime


-------------------------------------------------------------------------------
--[[                                 Types                                 ]]--
-------------------------------------------------------------------------------

local fid_t = Lc.legion_field_id_t

local LogicalRegion = {}
LogicalRegion.__index = LogicalRegion
L.LogicalRegion = LogicalRegion

local PhysicalRegion = {}
PhysicalRegion.__index = PhysicalRegion
L.PhysicalRegion = PhysicalRegion


-----------------------------------------------------------------------------
-- Module methods, to create logical regions, and allocate rows and fields --
-----------------------------------------------------------------------------

function LogicalRegion:AllocateRows(num)
  if self.rows_live + num > self.rows_max then
    error("Cannot allocate more rows for relation ", self.relation:Name())
  end
  Lc.legion_index_allocator_alloc(self.isa, num)
  self.rows_live = self.rows_live + num
end

-- NOTE: Assuming here that the compile time limit is never be hit.
function LogicalRegion:AllocateField(typ)
  local fid = Lc.legion_field_allocator_allocate_field(
                 self.fsa, terralib.sizeof(typ.terratype), self.field_ids)
  self.field_ids = self.field_ids + 1
  return fid
end

function L.NewLogicalRegion(params)
  local l = { rows_max  = params.rows_max,
              rows_live = 0,
              field_ids = 0,
              relation  = params.relation }
  l.is = Lc.legion_index_space_create(runtime, ctx, l.rows_max)
  l.fs = Lc.legion_field_space_create(runtime, ctx)
  l.isa = Lc.legion_index_allocator_create(runtime, ctx, l.is)
  l.fsa = Lc.legion_field_allocator_create(runtime, ctx, l.fs)
  l.handle = Lc.legion_logical_region_create(runtime, ctx, l.is, l.fs)
  setmetatable(l, LogicalRegion)
  l:AllocateRows(params.rows_init)
  return l
end


-------------------------------------------------------------------------------
--[[                    Privilege and coherence values                     ]]--
-------------------------------------------------------------------------------

L.privilege = {
  read_only     = Lc.READ_ONLY,
  read_write    = Lc.READ_WRITE,
  write_discard = Lc.WRITE_DISCARD,
  reduce        = Lc.REDUCE,
  default       = Lc.READ_WRITE
}

L.coherence = {
  exclusive    = Lc.EXCLUSIVE,
  atomic       = Lc.ATOMIC,
  simultaneous = Lc.SIMULTANEOUS,
  relaxed      = Lc.RELAXED,
  default      = Lc.EXCLUSIVE
}


-------------------------------------------------------------------------------
--[[              Create physical regions and task launchers               ]]--
-------------------------------------------------------------------------------

function LogicalRegion:CreatePhysicalRegion(params)
  local lreg = self.handle
  local privilege = params.privilege or L.privilege.default
  local coherence = params.coherence or L.coherence.default
  local input_launcher = Lc.legion_inline_launcher_create_logical_region(
                            lreg, privilege, coherence, lreg,
                            0, false, 0, 0)
  local fields = params.fields
  for i = 1, #fields do
    Lc.legion_inline_launcher_add_field(input_launcher, fields[i].fid, true)
  end
  local p = {}
  p.handle = Lc.legion_inline_launcher_execute(runtime, ctx, input_launcher)
  setmetatable(p, PhysicalRegion)
  return p
end

function PhysicalRegion:WaitUntilValid()
  Lc.legion_physical_region_wait_until_valid(self.handle)
end
