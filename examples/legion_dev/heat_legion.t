
local C   = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
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

local launch_task = macro(function(TASK_ID, arg, argtype, runtime, ctx)
  return quote
    var launcher = Lg.legion_task_launcher_create(
      TASK_ID,
      Lg.legion_task_argument_t {
        args    = &arg,
        arglen  = sizeof(argtype)
      },
      Lg.legion_predicate_true(),
      0, --Lg.legion_mapper_id_t,
      0  --Lg.legion_mapping_tag_id_t
    )
    var future = Lg.legion_task_launcher_execute(runtime,ctx,launcher)
    Lg.legion_task_launcher_destroy(launcher)
  in
    future
  end
end)

local Pt1dexplicit = macro(function(val)
  return `Lg.legion_point_1d_t { x = array(val) }
end) 
local Pt1d = macro(function(val)
  return `Lg.legion_domain_point_from_point_1d(
            Lg.legion_point_1d_t { x = array(val) })
end)


-- define the tasks

terra top_level_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  -- int num_elements = 1024; 
  -- // See if we have any command line arguments to parse
  -- {
  --   const InputArgs &command_args = HighLevelRuntime::get_input_args();
  --   for (int i = 1; i < command_args.argc; i++)
  --   {
  --     if (!strcmp(command_args.argv[i],"-n"))
  --       num_elements = atoi(command_args.argv[++i]);
  --   }
  -- }
  -- printf("Running daxpy for %d elements...\n", num_elements);
  var num_elements : int = 1024
  var command_args = Lg.legion_runtime_get_input_args()
  if command_args.argc > 1 then
    num_elements = C.atoi(command_args.argv[1])
    assert(num_elements > 0)
  end
  C.printf("Running daxpy for %d elements...\n", num_elements)

  -- TWO logical regions with common index space for input / output resp.
  -- Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  -- IndexSpace is = runtime->create_index_space(ctx, 
  --                         Domain::from_rect<1>(elem_rect));
  -- FieldSpace input_fs = runtime->create_field_space(ctx);
  -- {
  --   FieldAllocator allocator = 
  --     runtime->create_field_allocator(ctx, input_fs);
  --   allocator.allocate_field(sizeof(double),FID_X);
  --   allocator.allocate_field(sizeof(double),FID_Y);
  -- }
  -- FieldSpace output_fs = runtime->create_field_space(ctx);
  -- {
  --   FieldAllocator allocator = 
  --     runtime->create_field_allocator(ctx, output_fs);
  --   allocator.allocate_field(sizeof(double),FID_Z);
  -- }
  -- LogicalRegion input_lr =
  --        runtime->create_logical_region(ctx, is, input_fs);
  -- LogicalRegion output_lr =
  --        runtime->create_logical_region(ctx, is, output_fs);
  var elem_rect = Lg.legion_rect_1d_t {
    lo = Pt1dexplicit(0),
    hi = Pt1dexplicit(num_elements-1),
  }
  var is = Lg.legion_index_space_create_domain(runtime, ctx,
              Lg.legion_domain_from_rect_1d(elem_rect))
  var input_fs = Lg.legion_field_space_create(runtime, ctx)
  do
    var allocator = Lg.legion_field_allocator_create(runtime, ctx, input_fs)
    Lg.legion_field_allocator_allocate_field(allocator, sizeof(double), FID_X)
    Lg.legion_field_allocator_allocate_field(allocator, sizeof(double), FID_Y)
    Lg.legion_field_allocator_destroy(allocator)
  end
  var output_fs = Lg.legion_field_space_create(runtime, ctx)
  do
    var allocator = Lg.legion_field_allocator_create(runtime, ctx, output_fs)
    Lg.legion_field_allocator_allocate_field(allocator, sizeof(double), FID_Z)
    Lg.legion_field_allocator_destroy(allocator)
  end
  var input_lr = Lg.legion_logical_region_create(runtime, ctx, is, input_fs)
  var output_lr = Lg.legion_logical_region_create(runtime, ctx, is, output_fs)

  -- RegionRequirement req(input_lr, READ_WRITE, EXCLUSIVE, input_lr);
  -- req.add_field(FID_X);
  -- req.add_field(FID_Y);
  -- InlineLauncher input_launcher(req);
  var input_launcher = Lg.legion_inline_launcher_create_logical_region(
    input_lr,       -- legion_logical_region_t handle
    Lg.READ_WRITE,  -- legion_privilege_mode_t
    Lg.EXCLUSIVE,   -- legion_coherence_property_t
    input_lr,       -- legion_logical_region_t parent
    0,              -- legion_mapping_tag_id_t region_tag /* = 0 */
    false,          -- bool verified /* = false*/
    0,              -- legion_mapper_id_t id /* = 0 */
    0               -- legion_mapping_tag_id_t launcher_tag /* = 0 */
  )
  Lg.legion_inline_launcher_add_field(input_launcher, FID_X, true)
  Lg.legion_inline_launcher_add_field(input_launcher, FID_Y, true)

  -- PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
  -- input_region.wait_until_valid();
  var input_region =
    Lg.legion_inline_launcher_execute(runtime, ctx, input_launcher)
  Lg.legion_physical_region_wait_until_valid(input_region)
  -- Actually, it's not necessary to explicitly wait.
  -- The other API calls will implicitly do so for safety as needed

  -- RegionAccessor<AccessorType::Generic, double> acc_x = 
  --   input_region.get_field_accessor(FID_X).typeify<double>();
  -- RegionAccessor<AccessorType::Generic, double> acc_y = 
  --   input_region.get_field_accessor(FID_Y).typeify<double>();
  var acc_x =
    Lg.legion_physical_region_get_field_accessor_generic(input_region, FID_X)
  var acc_y =
    Lg.legion_physical_region_get_field_accessor_generic(input_region, FID_Y)

  -- for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  -- {
  --   acc_x.write(DomainPoint::from_point<1>(pir.p), drand48());
  --   acc_y.write(DomainPoint::from_point<1>(pir.p), drand48());
  -- }
  -- NOTE: WE WANT TO USE A DIFFERENT API
  do
    var subrect : Lg.legion_rect_1d_t
    var off_x   : Lg.legion_byte_offset_t
    var off_y   : Lg.legion_byte_offset_t
    var x_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_x, elem_rect, &subrect, &off_x))
    assert(x_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    var y_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_y, elem_rect, &subrect, &off_y))
    assert(y_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    for i=0,num_elements do
      var xptr = [&double](x_base + off_x.offset * i)
      var yptr = [&double](y_base + off_y.offset * i)
      @xptr = C.drand48()
      @yptr = C.drand48()
    end
  end

  -- InlineLauncher output_launcher(RegionRequirement(output_lr, WRITE_DISCARD,
  --                                                  EXCLUSIVE, output_lr));
  -- output_launcher.requirement.add_field(FID_Z);
  var output_launcher = Lg.legion_inline_launcher_create_logical_region(
    output_lr,        -- legion_logical_region_t handle
    Lg.WRITE_DISCARD, -- legion_privilege_mode_t
    Lg.EXCLUSIVE,     -- legion_coherence_property_t
    output_lr,        -- legion_logical_region_t parent
    0,                -- legion_mapping_tag_id_t region_tag /* = 0 */
    false,            -- bool verified /* = false*/
    0,                -- legion_mapper_id_t id /* = 0 */
    0                 -- legion_mapping_tag_id_t launcher_tag /* = 0 */
  )
  Lg.legion_inline_launcher_add_field(output_launcher, FID_Z, true)

  -- PhysicalRegion output_region = runtime->map_region(ctx, output_launcher);
  -- RegionAccessor<AccessorType::Generic, double> acc_z = 
  --   output_region.get_field_accessor(FID_Z).typeify<double>();
  var output_region =
    Lg.legion_inline_launcher_execute(runtime, ctx, output_launcher)
  var acc_z =
    Lg.legion_physical_region_get_field_accessor_generic(output_region, FID_Z)

  -- const double alpha = drand48();
  -- printf("Running daxpy computation with alpha %.8g...", alpha);
  -- for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  -- {
  --   double value = alpha * acc_x.read(DomainPoint::from_point<1>(pir.p)) + 
  --                          acc_y.read(DomainPoint::from_point<1>(pir.p));
  --   acc_z.write(DomainPoint::from_point<1>(pir.p), value);
  -- }
  -- printf("Done!\n");
  -- AGAIN, WE USE A DIFFERNT API
  var alpha : double = C.drand48()
  C.printf("Running daxpy computation with alpha %.8g...", alpha)
  do
    var subrect : Lg.legion_rect_1d_t
    var off_x   : Lg.legion_byte_offset_t
    var off_y   : Lg.legion_byte_offset_t
    var off_z   : Lg.legion_byte_offset_t
    var x_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_x, elem_rect, &subrect, &off_x))
    assert(x_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    var y_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_y, elem_rect, &subrect, &off_y))
    assert(y_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    var z_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_z, elem_rect, &subrect, &off_z))
    assert(z_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    for i=0,num_elements do
      var xptr = [&double](x_base + off_x.offset * i)
      var yptr = [&double](y_base + off_y.offset * i)
      var zptr = [&double](z_base + off_z.offset * i)
      var value = alpha * @xptr + @yptr
      @zptr = value
    end
  end
  C.printf("Done!\n")

  -- runtime->unmap_region(ctx, output_region);
  Lg.legion_runtime_unmap_region(runtime, ctx, output_region)

  -- output_launcher.requirement.privilege = READ_ONLY;
  -- output_region = runtime->map_region(ctx, output_launcher);
  Lg.legion_accessor_generic_destroy(acc_z)
  Lg.legion_physical_region_destroy(output_region)
  Lg.legion_inline_launcher_destroy(output_launcher)
  output_launcher = Lg.legion_inline_launcher_create_logical_region(
    output_lr,        -- legion_logical_region_t handle
    Lg.READ_ONLY,     -- legion_privilege_mode_t
    Lg.EXCLUSIVE,     -- legion_coherence_property_t
    output_lr,        -- legion_logical_region_t parent
    0,                -- legion_mapping_tag_id_t region_tag /* = 0 */
    false,            -- bool verified /* = false*/
    0,                -- legion_mapper_id_t id /* = 0 */
    0                 -- legion_mapping_tag_id_t launcher_tag /* = 0 */
  )
  Lg.legion_inline_launcher_add_field(output_launcher, FID_Z, true)
  output_region =
    Lg.legion_inline_launcher_execute(runtime, ctx, output_launcher)

  -- acc_z = output_region.get_field_accessor(FID_Z).typeify<double>();
  acc_z =
    Lg.legion_physical_region_get_field_accessor_generic(output_region, FID_Z)

  -- printf("Checking results...");
  -- bool all_passed = true;
  -- for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  -- {
  --   double expected = alpha * acc_x.read(DomainPoint::from_point<1>(pir.p))
  --                           + acc_y.read(DomainPoint::from_point<1>(pir.p));
  --   double received = acc_z.read(DomainPoint::from_point<1>(pir.p));
  --   if (expected != received)
  --     all_passed = false;
  -- }
  -- if (all_passed)
  --   printf("SUCCESS!\n");
  -- else
  --   printf("FAILURE!\n");
  C.printf("Checking results...")
  var all_passed : bool = true
  do
    var subrect : Lg.legion_rect_1d_t
    var off_x   : Lg.legion_byte_offset_t
    var off_y   : Lg.legion_byte_offset_t
    var off_z   : Lg.legion_byte_offset_t
    var x_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_x, elem_rect, &subrect, &off_x))
    assert(x_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    var y_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_y, elem_rect, &subrect, &off_y))
    assert(y_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    var z_base = [&int8](Lg.legion_accessor_generic_raw_rect_ptr_1d(
      acc_z, elem_rect, &subrect, &off_z))
    assert(z_base ~= nil)
    assert(elem_rect.lo.x[0] == subrect.lo.x[0] and
           elem_rect.hi.x[0] == subrect.hi.x[0])
    for i=0,num_elements do
      var xptr = [&double](x_base + off_x.offset * i)
      var yptr = [&double](y_base + off_y.offset * i)
      var zptr = [&double](z_base + off_z.offset * i)
      var expected = alpha * @xptr + @yptr
      var received = @zptr
      if expected ~= received then
        C.printf('x y expect z %.8g %.8g %.8g %.8g\n',
          @xptr, @yptr, expected, @zptr)
        all_passed = false
      end
    end
  end
  if all_passed then
    C.printf("SUCCESS!\n")
  else
    C.printf("FAILURE!\n")
  end

  -- additional cleanup necessary because of the C interface
  Lg.legion_accessor_generic_destroy(acc_z)
  Lg.legion_accessor_generic_destroy(acc_y)
  Lg.legion_accessor_generic_destroy(acc_x)
  Lg.legion_physical_region_destroy(output_region)
  Lg.legion_inline_launcher_destroy(output_launcher)
  Lg.legion_physical_region_destroy(input_region)
  Lg.legion_inline_launcher_destroy(input_launcher)

  -- // Clean up all our data structures.
  -- runtime->destroy_logical_region(ctx, input_lr);
  -- runtime->destroy_logical_region(ctx, output_lr);
  -- runtime->destroy_field_space(ctx, input_fs);
  -- runtime->destroy_field_space(ctx, output_fs);
  -- runtime->destroy_index_space(ctx, is);
  Lg.legion_logical_region_destroy(runtime, ctx, input_lr)
  Lg.legion_logical_region_destroy(runtime, ctx, output_lr)
  Lg.legion_field_space_destroy(runtime, ctx, input_fs)
  Lg.legion_field_space_destroy(runtime, ctx, output_fs)
  Lg.legion_index_space_destroy(runtime, ctx, is)


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

print('exit')