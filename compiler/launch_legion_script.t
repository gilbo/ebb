-- Launch liszt program as a top level legion task.

require('legionlib')

local function top_level_err_handler ( errobj )
  local err = tostring(errobj)
  if string.match(err, 'stack traceback:') then
    print(err)
  else
    print(err .. '\n' .. debug.traceback())
  end
  os.exit(1)
end

-- Error handler is copied from launch_script.t.
-- Not sure how legion handles errors.
-- If legion runtime prints stack trace,
-- we can remove error handling from here.
function top_level_task(binding, regions, args)
  local script_filename = args[1]
  success = xpcall( function ()
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
