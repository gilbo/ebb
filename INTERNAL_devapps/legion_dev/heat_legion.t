

local legion_dir    = './legion'
local bindings_dir  = legion_dir..'/bindings/terra'
local runtime_dir   = legion_dir..'/runtime'

-- Link in a particular library
terralib.linklibrary(bindings_dir..'/liblegion_terra.so')
-- Extend the Terra path that looks for C header files
terralib.includepath = terralib.includepath..';'..runtime_dir..
                                             ';'..bindings_dir
-- Extend the Lua path
package.path = package.path .. ';'..bindings_dir..'/?.t'


local C   = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

void flush() {
  fflush(stdout);
}
]])
local Lg  = terralib.includecstring([[
#include "legion_c.h"
]])


local TOP_LEVEL_TASK_ID       = 0
local compute_step_TASK_ID    = 1
local propagate_temp_TASK_ID  = 2
local clear_TASK_ID           = 3

local FID_position            = 0
local FID_head                = 1
local FID_tail                = 2
local FID_flux                = 3
local FID_jacobistep          = 4
local FID_temperature         = 5

-- helpers
local function terra2fp(tfunc)
  local def = tfunc:getdefinitions()[1]
  return def:getpointer()
end

local task_launcher = macro(function(TASK_ID, runtime, ctx)
  return `Lg.legion_task_launcher_create(
    TASK_ID,
    Lg.legion_task_argument_t {
      args    = nil,
      arglen  = 0
    },
    Lg.legion_predicate_true(),
    0, --Lg.legion_mapper_id_t,
    0  --Lg.legion_mapping_tag_id_t
  )
end)
local launch_task = macro(function(TASK_ID, runtime, ctx)
  return quote
    var launcher = task_launcher(TASK_ID, runtime, ctx)
    var future = Lg.legion_task_launcher_execute(runtime,ctx,launcher)
    Lg.legion_task_launcher_destroy(launcher)
  in
    future
  end
end)
local add_region = macro(function(launcher, lr)
  return `Lg.legion_task_launcher_add_region_requirement_logical_region(
    launcher,
    lr,
    Lg.READ_WRITE,
    Lg.EXCLUSIVE,
    lr,
    0,
    false)
end)

local Pt1dexplicit = macro(function(val)
  return `Lg.legion_point_1d_t { x = array(val) }
end) 
local Pt1d = macro(function(val)
  return `Lg.legion_domain_point_from_point_1d(
            Lg.legion_point_1d_t { x = array(val) })
end)
local Rect1d = macro(function(loval, hival)
  return `Lg.legion_rect_1d_t {
    lo = Pt1dexplicit(loval),
    hi = Pt1dexplicit(hival)
  }
end)
local RectDomain1d = macro(function(loval, hival)
  return `Lg.legion_domain_from_rect_1d(Rect1d(loval,hival))
end)


struct array_accessor {
  offset : Lg.legion_byte_offset_t,
  base   : &int8
}
local elem_ptr = macro(function(acc, i)
  return `(acc.base + acc.offset.offset * i)
end)
local get_accessor = macro(function(N_elem, pr, fid)
  return quote
    var lgacc = Lg.legion_physical_region_get_field_accessor_generic(pr, fid)
    var subrect : Lg.legion_rect_1d_t
    var arracc  : array_accessor
    arracc.base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      lgacc, Rect1d(0,N_elem-1), &subrect, &(arracc.offset)))
    Lg.legion_accessor_generic_destroy(lgacc)
  in
    arracc
  end
end)




-- define the tasks

terra top_level_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)

  -- describe a cube; # of indices
  var vertices_is = Lg.legion_index_space_create_domain(
    runtime, ctx, RectDomain1d(0,7)
  )
  var edges_is = Lg.legion_index_space_create_domain(
    runtime, ctx, RectDomain1d(0,11)
  )


  -- describe a cube; declare fields
  var vertices_fs = Lg.legion_field_space_create(runtime, ctx)
  do
    var allocator = Lg.legion_field_allocator_create(
      runtime, ctx, vertices_fs)
    Lg.legion_field_allocator_allocate_field(
      allocator, 3*sizeof(double), FID_position)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(float), FID_flux)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(float), FID_jacobistep)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(float), FID_temperature)
    Lg.legion_field_allocator_destroy(allocator)
  end
  var edges_fs = Lg.legion_field_space_create(runtime, ctx)
  do
    var allocator = Lg.legion_field_allocator_create(
      runtime, ctx, edges_fs)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), FID_head)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), FID_tail)
    Lg.legion_field_allocator_destroy(allocator)
  end

  -- describe a cube; create logical regions
  var vertices_lr =
    Lg.legion_logical_region_create(runtime, ctx, vertices_is, vertices_fs)
  var edges_lr =
    Lg.legion_logical_region_create(runtime, ctx, edges_is, edges_fs)


  -- INIT CUBE VERTEX POSITIONS TO BE and init other fields
  --[[
    0   0 0 0
    1   1 0 0
    2   1 1 0
    3   0 1 0
    4   0 0 1
    5   1 0 1
    6   1 1 1
    7   0 1 1
  ]]--
  var vertices_launcher = Lg.legion_inline_launcher_create_logical_region(
    vertices_lr,    -- legion_logical_region_t handle
    Lg.READ_WRITE,  -- legion_privilege_mode_t
    Lg.EXCLUSIVE,   -- legion_coherence_property_t
    vertices_lr,    -- legion_logical_region_t parent
    0,              -- legion_mapping_tag_id_t region_tag /* = 0 */
    false,          -- bool verified /* = false*/
    0,              -- legion_mapper_id_t id /* = 0 */
    0               -- legion_mapping_tag_id_t launcher_tag /* = 0 */
  )
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_position, true)
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_flux, true)
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_jacobistep, true)
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_temperature, true)
  var vertices_pr =
    Lg.legion_inline_launcher_execute(runtime, ctx, vertices_launcher)

  var posacc = get_accessor(8, vertices_pr, FID_position)
  @[&double[3]](elem_ptr(posacc,0)) = arrayof(double,0,0,0)
  @[&double[3]](elem_ptr(posacc,1)) = arrayof(double,1,0,0)
  @[&double[3]](elem_ptr(posacc,2)) = arrayof(double,1,1,0)
  @[&double[3]](elem_ptr(posacc,3)) = arrayof(double,0,1,0)
  @[&double[3]](elem_ptr(posacc,4)) = arrayof(double,0,0,1)
  @[&double[3]](elem_ptr(posacc,5)) = arrayof(double,1,0,1)
  @[&double[3]](elem_ptr(posacc,6)) = arrayof(double,1,1,1)
  @[&double[3]](elem_ptr(posacc,7)) = arrayof(double,0,1,1)
  var fluxacc = get_accessor(8, vertices_pr, FID_flux)
  var jacacc  = get_accessor(8, vertices_pr, FID_jacobistep)
  var temacc  = get_accessor(8, vertices_pr, FID_temperature)
  for i=0,8 do
    @[&float](elem_ptr(fluxacc,i)) = 0
    @[&float](elem_ptr(jacacc, i)) = 0
    @[&float](elem_ptr(temacc, i)) = 0
  end
  @[&float](elem_ptr(temacc,0)) = 1000

  Lg.legion_runtime_unmap_region(runtime, ctx, vertices_pr)
  Lg.legion_physical_region_destroy(vertices_pr)
  Lg.legion_inline_launcher_destroy(vertices_launcher)


  -- INIT CUBE EDGE TOPOLOGY TO BE
  --[[  head/tail
    0   1 0
    1   2 1
    2   3 2
    3   0 3
    4   5 4
    5   6 5
    6   7 6
    7   4 7
    8   5 1
    9   0 4
    10  6 2
    11  7 3
  ]]--
  var edges_launcher = Lg.legion_inline_launcher_create_logical_region(
    edges_lr,       -- legion_logical_region_t handle
    Lg.READ_WRITE,  -- legion_privilege_mode_t
    Lg.EXCLUSIVE,   -- legion_coherence_property_t
    edges_lr,       -- legion_logical_region_t parent
    0,              -- legion_mapping_tag_id_t region_tag /* = 0 */
    false,          -- bool verified /* = false*/
    0,              -- legion_mapper_id_t id /* = 0 */
    0               -- legion_mapping_tag_id_t launcher_tag /* = 0 */
  )
  Lg.legion_inline_launcher_add_field(edges_launcher, FID_head, true)
  Lg.legion_inline_launcher_add_field(edges_launcher, FID_tail, true)
  var edges_pr =
    Lg.legion_inline_launcher_execute(runtime, ctx, edges_launcher)

  var headacc = get_accessor(12, edges_pr, FID_head)
  @[&uint64](elem_ptr(headacc,0))  = 1
  @[&uint64](elem_ptr(headacc,1))  = 2
  @[&uint64](elem_ptr(headacc,2))  = 3
  @[&uint64](elem_ptr(headacc,3))  = 0
  @[&uint64](elem_ptr(headacc,4))  = 5
  @[&uint64](elem_ptr(headacc,5))  = 6
  @[&uint64](elem_ptr(headacc,6))  = 7
  @[&uint64](elem_ptr(headacc,7))  = 4
  @[&uint64](elem_ptr(headacc,8))  = 5
  @[&uint64](elem_ptr(headacc,9))  = 0
  @[&uint64](elem_ptr(headacc,10)) = 6
  @[&uint64](elem_ptr(headacc,11)) = 7

  var tailacc = get_accessor(12, edges_pr, FID_tail)
  @[&uint64](elem_ptr(tailacc,0))  = 0
  @[&uint64](elem_ptr(tailacc,1))  = 1
  @[&uint64](elem_ptr(tailacc,2))  = 2
  @[&uint64](elem_ptr(tailacc,3))  = 3
  @[&uint64](elem_ptr(tailacc,4))  = 4
  @[&uint64](elem_ptr(tailacc,5))  = 5
  @[&uint64](elem_ptr(tailacc,6))  = 6
  @[&uint64](elem_ptr(tailacc,7))  = 7
  @[&uint64](elem_ptr(tailacc,8))  = 1
  @[&uint64](elem_ptr(tailacc,9))  = 4
  @[&uint64](elem_ptr(tailacc,10)) = 2
  @[&uint64](elem_ptr(tailacc,11)) = 3

  Lg.legion_runtime_unmap_region(runtime, ctx, edges_pr)
  Lg.legion_physical_region_destroy(edges_pr)
  Lg.legion_inline_launcher_destroy(edges_launcher)


  -- LOOP THE TASKS 1000 TIMES
    -- loop body
  var launch_comp_step = task_launcher(compute_step_TASK_ID,    runtime, ctx)
  var launch_prop_temp = task_launcher(propagate_temp_TASK_ID,  runtime, ctx)
  var launch_clear     = task_launcher(clear_TASK_ID,           runtime, ctx)

  var eidx = add_region(launch_comp_step, edges_lr)
  var vidx = add_region(launch_comp_step, vertices_lr)
  Lg.legion_task_launcher_add_field(launch_comp_step, eidx, FID_head, true)
  Lg.legion_task_launcher_add_field(launch_comp_step, eidx, FID_tail, true)
  Lg.legion_task_launcher_add_field(
    launch_comp_step, vidx, FID_position, true)
  Lg.legion_task_launcher_add_field(
    launch_comp_step, vidx, FID_temperature, true)
  Lg.legion_task_launcher_add_field(
    launch_comp_step, vidx, FID_flux, true)
  Lg.legion_task_launcher_add_field(
    launch_comp_step, vidx, FID_jacobistep, true)

  vidx = add_region(launch_prop_temp, vertices_lr)
  Lg.legion_task_launcher_add_field(
    launch_prop_temp, vidx, FID_temperature, true)
  Lg.legion_task_launcher_add_field(
    launch_prop_temp, vidx, FID_flux, true)
  Lg.legion_task_launcher_add_field(
    launch_prop_temp, vidx, FID_jacobistep, true)

  vidx = add_region(launch_clear, vertices_lr)
  Lg.legion_task_launcher_add_field(launch_clear, vidx, FID_flux, true)
  Lg.legion_task_launcher_add_field(launch_clear, vidx, FID_jacobistep, true)

  for i=0,1000 do
    Lg.legion_task_launcher_execute(runtime,ctx,launch_comp_step)
    Lg.legion_task_launcher_execute(runtime,ctx,launch_prop_temp)
    Lg.legion_task_launcher_execute(runtime,ctx,launch_clear)
  end

  Lg.legion_task_launcher_destroy(launch_clear)
  Lg.legion_task_launcher_destroy(launch_prop_temp)
  Lg.legion_task_launcher_destroy(launch_comp_step)


  -- PRINT the temperature
  vertices_launcher = Lg.legion_inline_launcher_create_logical_region(
    vertices_lr,    -- legion_logical_region_t handle
    Lg.READ_WRITE,  -- legion_privilege_mode_t
    Lg.EXCLUSIVE,   -- legion_coherence_property_t
    vertices_lr,    -- legion_logical_region_t parent
    0,              -- legion_mapping_tag_id_t region_tag /* = 0 */
    false,          -- bool verified /* = false*/
    0,              -- legion_mapper_id_t id /* = 0 */
    0               -- legion_mapping_tag_id_t launcher_tag /* = 0 */
  )
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_position, true)
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_flux, true)
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_jacobistep, true)
  Lg.legion_inline_launcher_add_field(vertices_launcher, FID_temperature, true)
  vertices_pr =
    Lg.legion_inline_launcher_execute(runtime, ctx, vertices_launcher)

  temacc  = get_accessor(8, vertices_pr, FID_temperature)
  for i=0,8 do
    C.printf('%d %.8g\n', i, @[&float](elem_ptr(temacc, i)))
  end

  Lg.legion_runtime_unmap_region(runtime, ctx, vertices_pr)
  Lg.legion_physical_region_destroy(vertices_pr)
  Lg.legion_inline_launcher_destroy(vertices_launcher)


  -- CLEANUP
  Lg.legion_logical_region_destroy(runtime, ctx, edges_lr)
  Lg.legion_logical_region_destroy(runtime, ctx, vertices_lr)
  Lg.legion_field_space_destroy(runtime, ctx, edges_fs)
  Lg.legion_field_space_destroy(runtime, ctx, vertices_fs)
  Lg.legion_index_space_destroy(runtime, ctx, edges_is)
  Lg.legion_index_space_destroy(runtime, ctx, vertices_is)

end

terra compute_step(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  assert(n_regions == 2)
  var headacc = get_accessor(12, regions[0], FID_head)
  var tailacc = get_accessor(12, regions[0], FID_tail)
  var posacc  = get_accessor(8, regions[1], FID_position)
  var fluxacc = get_accessor(8, regions[1], FID_flux)
  var jacacc  = get_accessor(8, regions[1], FID_jacobistep)
  var temacc  = get_accessor(8, regions[1], FID_temperature)

  for i=0,12 do
    var v1 = @[&uint64](elem_ptr(headacc,i))
    var v2 = @[&uint64](elem_ptr(tailacc,i))

    var p1 = @[&double[3]](elem_ptr(posacc,v1))
    var p2 = @[&double[3]](elem_ptr(posacc,v2))
    var dp = array(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    var dt = @[&float](elem_ptr(temacc,v1)) - @[&float](elem_ptr(temacc,v2))
    var dplen = C.sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2])
    var step  = 1.0 / dplen

    var f1 = [&float](elem_ptr(fluxacc,v1))
    var f2 = [&float](elem_ptr(fluxacc,v2))
    @f1 = @f1 + -dt * step
    @f2 = @f2 +  dt * step

    var j1 = [&float](elem_ptr(jacacc,v1))
    var j2 = [&float](elem_ptr(jacacc,v2))
    @j1 = @j1 + step
    @j2 = @j2 + step
  end
end

terra propagate_temp(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  assert(n_regions == 1)
  var fluxacc = get_accessor(8, regions[0], FID_flux)
  var jacacc  = get_accessor(8, regions[0], FID_jacobistep)
  var temacc  = get_accessor(8, regions[0], FID_temperature)

  for i=0,8 do
    var t = [&float](elem_ptr(temacc,i))
    @t = @t +
      .01f * @[&float](elem_ptr(fluxacc,i)) / @[&float](elem_ptr(jacacc,i))
  end
end

terra clear(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  assert(n_regions == 1)
  var fluxacc = get_accessor(8, regions[0], FID_flux)
  var jacacc  = get_accessor(8, regions[0], FID_jacobistep)

  for i=0,8 do
    @[&float](elem_ptr(fluxacc,i)) = 0
    @[&float](elem_ptr(jacacc,i))  = 0
  end
end


function main()
  -- must first set the top level task ID
  Lg.legion_runtime_set_top_level_task_id(TOP_LEVEL_TASK_ID)

  -- Before starting the runtime, we need to
  -- register all possible tasks to launch with the runtime.

  -- Register Tasks
  local default_options = global(Lg.legion_task_config_options_t)
  default_options.leaf = false
  default_options.inner = false
  default_options.idempotent = false
  Lg.legion_runtime_register_task_void(
    TOP_LEVEL_TASK_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'top_level_task',
    terra2fp(top_level_task) --Lg.legion_task_pointer_t
  )
  Lg.legion_runtime_register_task_void(
    compute_step_TASK_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'compute_step',
    terra2fp(compute_step) --Lg.legion_task_pointer_t
  )
  Lg.legion_runtime_register_task_void(
    propagate_temp_TASK_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'propagate_temp',
    terra2fp(propagate_temp) --Lg.legion_task_pointer_t
  )
  Lg.legion_runtime_register_task_void(
    clear_TASK_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'clear',
    terra2fp(clear) --Lg.legion_task_pointer_t
  )

  -- WARNING: THIS FUNCTION LEAKS MEMORY
  local function arg_convert(args)
    local arguments = global(&&int8, C.malloc(#arg * sizeof(&int8)))
    for i=1,#arg do
      arguments:get()[i-1] = global(&int8, arg[i]):get()
    end
    return arguments:get()
  end

  -- Start the runtime
  Lg.legion_runtime_start(#arg,arg_convert(arg),false)
  -- false means don't run in the background
end

main()
