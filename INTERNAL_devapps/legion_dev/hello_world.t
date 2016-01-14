
local C   = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
]])
local Lg  = terralib.includecstring([[
#include "legion_c.h"
]])

--C.printf('Hello, World!\n')
--for k,v in pairs(Lg) do print(k,v) end


local HELLO_WORLD_ID = 0


-- define the task

terra hello_world_task(
  task      :  (Lg.legion_task_t),
  regions   : &(Lg.legion_physical_region_t),
  n_regions : uint,
  ctx       :  (Lg.legion_context_t),
  runtime   :  (Lg.legion_runtime_t)
)
  C.printf('Hello, World!\n')
  C.printf('foo %d\n', 1+2)
end
hello_world_task:compile()
local hw_def = hello_world_task:getdefinitions()[1]

function main()
  -- must first set the top level task ID
  --HighLevelRuntime::set_top_level_task_id(HELLO_WORLD_ID)
  Lg.legion_runtime_set_top_level_task_id(HELLO_WORLD_ID)

  -- Before starting the runtime, we need to
  -- register all possible tasks to launch with the runtime.

  local default_options = global(Lg.legion_task_config_options_t)
  default_options.leaf = false
  default_options.inner = false
  default_options.idempotent = false

  -- two booleans here are single=true and index=false
  -- HighLevelRuntime::register_legion_task(hello_world_task,HELLO_WORLD_ID, LgHL.Processor.LOC_PROC, true, false)
  Lg.legion_runtime_register_task_void(
    HELLO_WORLD_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'hello_world_task',
    hw_def:getpointer() --Lg.legion_task_pointer_t
  )

  -- build an argument structure to call the runtime with
  local arguments = global(&&int8, C.malloc(sizeof(&int8)*#arg))
  for i=1,#arg do
    arguments[i-1] = arg[i]
  end

  -- Start the runtime
  print('num args', #arg)
  Lg.legion_runtime_start(#arg,arguments:get(),false)
  -- false means don't run in the background
  -- HighLevelRuntime::start(argc,argv)
end

main()

-- the program will never get here btw
print('exit')

