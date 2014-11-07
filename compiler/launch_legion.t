-- Launch liszt program as a top level legion task.

terralib.require 'legionlib-terra'

-- Error handler used in top level task for meaningful traceback on errors.
-- This function is copied from launch_script.t.
local function top_level_err_handler ( errobj )
  local err = tostring(errobj)
  if string.match(err, 'stack traceback:') then
    print(err)
  else
    print(err .. '\n' .. debug.traceback())
  end
  os.exit(1)
end

-- Setup tables for Legion runtime in Liszt.
local function setup_for_legion(binding)
  local L = terralib.require 'compiler.lisztlib'
  L._runtime = L._Legion
  L._runtime.binding = binding
  local type_map = {
                     [L.float] = 'float',
                     [L.double] = 'double',
                     [L.int] = 'int',
                     [L.bool] = 'int'
                   }
  L._LegionTypes = {}
  for k, v in ipairs(type_map) do
    L._LegionTypes[k] = PrimType[v]
  end
end

function top_level_task(binding, regions, args)
  setup_for_legion(binding)
  -- launch application
  local script_filename = args[1]
  local success = xpcall( function ()
    assert(terralib.loadfile(script_filename))()
  end, top_level_err_handler)
  if success then
    os.exit(0)
  else
    os.exit(1)
  end
end

TOP_LEVEL = 100

if rawget(_G, "arg") then
    local binding = LegionLib:init_binding(arg[0])
    binding:set_top_level_task_id(TOP_LEVEL)
    binding:register_single_task(TOP_LEVEL, "top_level_task",
                                 Processor.LOC_PROC, false)
    binding:start(arg)
end
