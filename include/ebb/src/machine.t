-- The MIT License (MIT)
-- 
-- Copyright (c) 2016 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.


local Exports = {}
package.loaded["ebb.src.machine"] = Exports

local use_legion = not not rawget(_G, '_legion_env')
local use_exp    = not not rawget(_G, 'EBB_USE_EXPERIMENTAL_SIGNAL')
local use_single = not use_legion and not use_exp

local LE, legion_env, LW, use_partitioning
if use_legion then
  LE                = rawget(_G, '_legion_env')
  legion_env        = LE.legion_env[0]
  LW                = require 'ebb.src.legionwrap'
  use_partitioning  = rawget(_G, '_run_config').use_partitioning
end

local ewrap         = use_exp and require 'ebb.src.ewrap'

local Util              = require 'ebb.src.util'
local C                 = require 'ebb.src.c'
local ffi               = require 'ffi'

------------------------------------------------------------------------------

local newlist = terralib.newlist

-- a 64-bit value converted to hex should be...
local hex_str_buffer = global(int8[24])
local terra do_hex_conversion( val : uint64 ) : &int8
  C.snprintf(hex_str_buffer, 24, "%lx", val)
  return hex_str_buffer
end
local function tohexstr(obj)
  return ffi.string(do_hex_conversion(obj),16) -- exactly 16*4=64 bits
end

-------------------------------------------------------------------------------
--[[ Node Types:                                                           ]]--
-------------------------------------------------------------------------------

-- would be nice to define these here...
local NodeType    = {}
NodeType.__index  = NodeType

local all_node_types = newlist()

-- I expect to add more parameters and details here
-- as we get more specific about how we want to model the machine
local CreateNodeType = Util.memoize_named({
  'n_cpus', 'n_gpus',
},function(args)
  local nt = setmetatable({
    n_cpus = args.n_cpus,
    n_gpus = args.n_gpus,
  }, NodeType)
  all_node_types:insert(nt)
  return nt
end)

function Exports.GetAllNodeTypes() return all_node_types end

---- a simple default node type that should always work
---- though it severely under-estimates the compute power of a node
--local SingleCPUNode = setmetatable({},NodeType)
--Exports.SingleCPUNode = SingleCPUNode
--

-------------------------------------------------------------------------------
--[[ Machine Setup / Detection:                                            ]]--
-------------------------------------------------------------------------------


local extract_all_processors, extract_all_memories = true, true
local group_nodes = true

if use_legion then

-- from legion headers, a processor kind is one of the following enumeration
--[[
    typedef enum legion_lowlevel_processor_kind_t {
      NO_KIND,
      TOC_PROC,     // Throughput core
      LOC_PROC,     // Latency core
      UTIL_PROC,    // Utility core
      IO_PROC,      // I/O core
      PROC_GROUP,   // Processor group
    } legion_lowlevel_processor_kind_t;
--]]
local proc_kind_str = {
  [LW.TOC_PROC]   = 'GPU',
  [LW.LOC_PROC]   = 'CPU',
  [LW.UTIL_PROC]  = 'UPU',
  -- otherwise we ignore it I guess...
}

-- from from legion headers, a memory kind is one of these
--[[
    typedef enum legion_lowlevel_memory_kind_t {
      GLOBAL_MEM, // Guaranteed visible to all processors on all nodes (e.g. GASNet memory, universally slow)
      SYSTEM_MEM, // Visible to all processors on a node
      REGDMA_MEM, // Registered memory visible to all processors on a node, can be a target of RDMA
      SOCKET_MEM, // Memory visible to all processors within a node, better performance to processors on same socket 
      Z_COPY_MEM, // Zero-Copy memory visible to all CPUs within a node and one or more GPUs 
      GPU_FB_MEM,   // Framebuffer memory for one GPU and all its SMs
      DISK_MEM,   // Disk memory visible to all processors on a node
      HDF_MEM,    // HDF memory visible to all processors on a node
      FILE_MEM,   // file memory visible to all processors on a node
      LEVEL3_CACHE, // CPU L3 Visible to all processors on the node, better performance to processors on same socket 
      LEVEL2_CACHE, // CPU L2 Visible to all processors on the node, better performance to one processor
      LEVEL1_CACHE, // CPU L1 Visible to all processors on the node, better performance to one processor
    } legion_lowlevel_memory_kind_t;
--]]
local mem_kind_str = {
  --[LW.GLOBAL_MEM]     = 'GLOBAL_MEM',
  [LW.SYSTEM_MEM]     = 'SYSTEM_MEM',
  [LW.GPU_FB_MEM]     = 'GPU_FB_MEM',
  -- otherwise, meh
}

assert(extract_all_processors)
function extract_all_processors( machine )
  local n_proc  = tonumber(LW.legion_machine_get_all_processors_size(machine))
  local procs   = terralib.cast( &LW.legion_processor_t,
                      C.malloc(sizeof(LW.legion_processor_t) * n_proc) )
  LW.legion_machine_get_all_processors(machine, procs, n_proc)

  local ps = newlist()
  for i=0,(n_proc-1) do
    local pobj  = procs[i]
    local ptyp  = proc_kind_str[ LW.legion_processor_kind(pobj) ]
    local paddr = LW.legion_processor_address_space(pobj) -- is a uint
    if ptyp then
      ps:insert {
        id    = pobj.id,
        type  = ptyp,
        addr  = paddr,
      }
    end
  end

  --for _,p in ipairs(ps) do
  --  print(tohexstr(p.id), p.type, p.addr)
  --end

  return ps
end

local function extract_all_memories( machine )
  local n_mems  = tonumber(LW.legion_machine_get_all_memories_size(machine))
  local memarr  = terralib.cast( &LW.legion_memory_t,
                      C.malloc(sizeof(LW.legion_memory_t) * n_mems) )
  LW.legion_machine_get_all_memories(machine, memarr, n_mems)

  local mems = newlist()
  for i=0,(n_mems-1) do
    local mobj  = memarr[i]
    local mtyp  = mem_kind_str[ LW.legion_memory_kind(mobj) ]
    local maddr = LW.legion_memory_address_space(mobj) -- is a uint
    --print(tohexstr(mobj.id), LW.legion_memory_kind(mobj), maddr)
    if mtyp then
      mems:insert {
        id    = mobj.id,
        type  = mtyp,
        addr  = maddr,
      }
    end
  end

  --for _,m in ipairs(mems) do
  --  print(tohexstr(m.id), m.type, m.addr)
  --end

  return mems
end

-- Group nodes based on address space
local function group_nodes(procs, mems)
  local node_map = {}
  local function get_node(addr_space)
    if not node_map[addr_space] then
      node_map[addr_space] = {
        addr = addr_space, procs=newlist(), mems=newlist()
      }
    end
    return node_map[addr_space]
  end

  for _,p in ipairs(procs) do   get_node(p.addr).procs:insert(p)  end
  for _,m in ipairs(mems) do    get_node(m.addr).procs:insert(m)  end

  -- analyze on a per-node basis
  local nodes = newlist()
  for _,node in pairs(node_map) do
    node.cpus = newlist()
    node.gpus = newlist()
    for _,p in ipairs(node.procs) do
      if      p.type == 'CPU' then node.cpus:insert(p)
      elseif  p.type == 'GPU' then node.gpus:insert(p) end
    end
    -- types are memoized, so this will collapse the total number of
    -- node types
    node.node_type = CreateNodeType{
      n_cpus = #node.cpus,
      n_gpus = #node.gpus,
    }
    nodes:insert(node)
  end
end

end -- end use_legion



-------------------------------------------------------------------------------
--[[ Machine Setup / Detection:                                            ]]--
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
--[[ Defining a Machine                                                    ]]--
-------------------------------------------------------------------------------

local Machine   = {}
Machine.__index = Machine
local TheMachine

if use_legion and use_partitioning then
  local function initialize_legion_machine_model()
    local machine = LW.legion_machine_create()
    
    local procs = extract_all_processors(machine)
    local mems  = extract_all_memories(machine)

    local nodes = group_nodes(procs, mems)

    TheMachine = setmetatable({
      nodes = nodes,
    }, Machine)

    --print('OK\nOK\nOK\nOK')

    LW.legion_machine_destroy(machine)
  end

  initialize_legion_machine_model()
end

if use_exp then
  local function initialize_machine_model()
    local n_nodes     = ewrap.N_NODES
    local this_node   = ewrap.THIS_NODE

    -- for now just assume each machine has one CPU
    local nodes = newlist()
    for i=1,n_nodes do
      local n_threads = 1

      local node = {
        node_id = i-1,
        node_type = CreateNodeType{
          n_cpus = 1,
          n_gpus = 0,
        },
      }
    end

    TheMachine = setmetatable({
      nodes = nodes,
    }, Machine)
  end

  initialize_machine_model()
end











-------------------------------------------------------------------------------
--[[ Example Code from Legion Team (for reference)                         ]]--
-------------------------------------------------------------------------------

--[[
local terra get_n_proc( machine : LW.legion_machine_t )
  var query   = LW.legion_processor_query_create(machine)
  var n_proc  = LW.legion_processor_query_count(query)
  LW.legion_processor_query_destroy(query)
  return n_proc
end

local terra get_n_mems( machine : LW.legion_machine_t )
  var query   = LW.legion_memory_query_create(machine)
  var n_mems  = LW.legion_memory_query_count(query)
  LW.legion_memory_query_destroy(query)
  return n_mems
end

local function proc_query( machine, proc_type, proc_type_name )
  local procs = newlist()

  local query   = LW.legion_processor_query_create(machine)
                  LW.legion_processor_query_only_kind(query, proc_type)
  local n_procs = tonumber(LW.legion_processor_query_count(query))

  local proc    = LW.legion_processor_query_first(query)
  while proc.id ~= 0 do
    local memlist = newlist()
    procs:insert {
      id    = proc.id,
      type  = proc_type_name,
      mems  = memlist, -- memories?
    }

    -- record all the addressable memories
    local memquery  = LW.legion_memory_query_create(machine)
    LW.legion_memory_query_has_affinity_to_processor(memquery, proc, 0, 0)
    local n_mems    = LW.legion_memory_query_count(memquery)
    local mem       = LW.legion_memory_query_first(memquery)
    while mem.id ~= 0 do
      memlist:insert(mem.id)
      mem = LW.legion_memory_query_next(memquery, mem)
    end
    LW.legion_memory_query_destroy(memquery)
    proc  = LW.legion_processor_query_next(query, proc)
  end
  LW.legion_processor_query_destroy(query)


  return procs
end

local function print_proc_query(procs, name)
  print("  "..#procs.." "..name..":")
  for _,p in pairs(procs) do
    print("    "..tohexstr(p.id).." addresses "..#p.mems.." memories:")
    for _,id in ipairs(p.mems) do print("      "..tohexstr(id)) end
  end
end



local terra memqueries( machine : LW.legion_machine_t )
  -- Query memories
  var global_mems = LW.legion_memory_query_create(machine)
  LW.legion_memory_query_only_kind(global_mems, LW.GLOBAL_MEM)
  C.printf("  %lu Global:\n", LW.legion_memory_query_count(global_mems))
  var global_mem = LW.legion_memory_query_first(global_mems)
  while global_mem.id ~= 0 do
    C.printf("    %lx\n", global_mem.id)
    global_mem = LW.legion_memory_query_next(global_mems, global_mem)
  end
  LW.legion_memory_query_destroy(global_mems)
  var sys_mems = LW.legion_memory_query_create(machine)
  LW.legion_memory_query_only_kind(sys_mems, LW.SYSTEM_MEM)
  C.printf("  %lu System:\n", LW.legion_memory_query_count(sys_mems))
  var sys_mem = LW.legion_memory_query_first(sys_mems)
  while sys_mem.id ~= 0 do
    var proc_affinity = LW.legion_processor_query_create(machine)
    LW.legion_processor_query_best_affinity_to_memory(proc_affinity, sys_mem, 0, 0)
    C.printf("    %lx with affinity to %lu memories:\n", sys_mem.id, LW.legion_processor_query_count(proc_affinity))
    var proc = LW.legion_processor_query_first(proc_affinity)
    while proc.id ~= 0 do
      C.printf("      %lx\n", proc.id)
      proc = LW.legion_processor_query_next(proc_affinity, proc)
    end
    LW.legion_processor_query_destroy(proc_affinity)
    sys_mem = LW.legion_memory_query_next(sys_mems, sys_mem)
  end
  LW.legion_memory_query_destroy(sys_mems)
  var reg_mems = LW.legion_memory_query_create(machine)
  LW.legion_memory_query_only_kind(reg_mems, LW.REGDMA_MEM)
  C.printf("  %lu Registered:\n", LW.legion_memory_query_count(reg_mems))
  var reg_mem = LW.legion_memory_query_first(reg_mems)
  while reg_mem.id ~= 0 do
    C.printf("    %lx\n", reg_mem.id)
    reg_mem = LW.legion_memory_query_next(reg_mems, reg_mem)
  end
  LW.legion_memory_query_destroy(reg_mems)
  var zcopy_mems = LW.legion_memory_query_create(machine)
  LW.legion_memory_query_only_kind(zcopy_mems, LW.Z_COPY_MEM)
  C.printf("  %lu Zero Copy:\n", LW.legion_memory_query_count(zcopy_mems))
  var zcopy_mem = LW.legion_memory_query_first(zcopy_mems)
  while zcopy_mem.id ~= 0 do
    C.printf("    %lx\n", zcopy_mem.id)
    zcopy_mem = LW.legion_memory_query_next(zcopy_mems, zcopy_mem)
  end
  LW.legion_memory_query_destroy(zcopy_mems)
  var fb_mems = LW.legion_memory_query_create(machine)
  LW.legion_memory_query_only_kind(fb_mems, LW.GPU_FB_MEM)
  C.printf("  %lu Frame Buffer:\n", LW.legion_memory_query_count(fb_mems))
  var fb_mem = LW.legion_memory_query_first(fb_mems)
  while fb_mem.id ~= 0 do
    C.printf("    %lx\n", fb_mem.id)
    fb_mem = LW.legion_memory_query_next(fb_mems, fb_mem)
  end
  LW.legion_memory_query_destroy(fb_mems)
end


local function dummy_query()
  local machine = LW.legion_machine_create()

  -- Query processors
  C.printf("\nTotal processors: %lu\n",get_n_proc(machine))
  C.printf("Total memories:   %lu\n",get_n_mems(machine))

  local cpus = proc_query(machine, LW.LOC_PROC)
  local gpus = proc_query(machine, LW.TOC_PROC)
  local upus = proc_query(machine, LW.UTIL_PROC)
  print_proc_query(cpus,'CPU(s)','CPU')
  print_proc_query(gpus,'GPU(s)','GPU')
  print_proc_query(upus,'Utility Processor(s)','UPU')


  memqueries(machine)

  LW.legion_machine_destroy(machine)
end
--]]


