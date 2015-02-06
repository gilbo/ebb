
local C   = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
]])
local Lg  = terralib.includecstring([[
#include "legion_c.h"
]])


local TOP_LEVEL_TASK_ID     = 0

local FID_FIELD_A           = 0
local FID_FIELD_B           = 1

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
  -- IndexSpace unstructured_is = runtime->create_index_space(ctx, 1024); 
  -- printf("Created unstructured index space %x\n", unstructured_is.id);
  -- Rect<1> rect(Point<1>(0),Point<1>(1023));
  -- IndexSpace structured_is = runtime->create_index_space(ctx, 
  --                                         Domain::from_rect<1>(rect));
  -- printf("Created structured index space %x\n", structured_is.id);
  var unstructured_is = Lg.legion_index_space_create(runtime, ctx, 1024)
  -- 1024 is the maximum # of elements (not the actual number of elements)
  C.printf("Created unstructured index space %x\n", unstructured_is.id)
  var rect = Lg.legion_rect_1d_t {
    lo = Pt1dexplicit(0),
    hi = Pt1dexplicit(1023),
  }
  var structured_is = Lg.legion_index_space_create_domain(
    runtime, ctx, Lg.legion_domain_from_rect_1d(rect)
  )
  C.printf("Created structured index space %x\n", structured_is.id)

  -- {
  --   IndexAllocator allocator = runtime->create_index_allocator(ctx, 
  --                                                   unstructured_is);
  --   ptr_t begin = allocator.alloc(1024);
  --   // Make sure it isn't null
  --   assert(!begin.is_null());
  --   printf("Allocated elements in unstructured "
  --          "space at ptr_t %d\n", begin.value);
  --   // When the allocator goes out of scope the runtime reclaims
  --   // its resources.
  -- }
  do
    var allocator =
      Lg.legion_index_allocator_create(runtime, ctx, unstructured_is)
    var begin = Lg.legion_index_allocator_alloc(allocator, 1024)
    assert(not Lg.legion_ptr_is_null(begin))
    C.printf("Allocated elements in unstructured space at ptr_t %d\n",
             begin.value)
    Lg.legion_index_allocator_destroy(allocator)
  end

  -- {
  --   Domain orig_domain =
  --     runtime->get_index_space_domain(ctx, structured_is);
  --   Rect<1> orig = orig_domain.get_rect<1>();
  --   assert(orig == rect);
  -- }
  do
    var orig_domain =
      Lg.legion_index_space_get_domain(runtime, ctx, structured_is)
    var orig = Lg.legion_domain_get_rect_1d(orig_domain)
    assert(orig.lo.x[0] == rect.lo.x[0] and orig.hi.x[0] == rect.hi.x[0])
  end

  -- FieldSpace fs = runtime->create_field_space(ctx);
  -- printf("Created field space field space %x\n", fs.get_id());
  -- {
  --   FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  --   FieldID fida = allocator.allocate_field(sizeof(double), FID_FIELD_A);
  --   assert(fida == FID_FIELD_A);
  --   FieldID fidb = allocator.allocate_field(sizeof(int), FID_FIELD_B);
  --   assert(fidb == FID_FIELD_B);
  --   printf("Allocated two fields with Field IDs %d and %d\n", fida, fidb);
  -- }
  var fs = Lg.legion_field_space_create(runtime, ctx)
  C.printf("Created field space field space %x\n", fs.id)
  do
    var allocator = Lg.legion_field_allocator_create(runtime, ctx, fs)
    var fida = Lg.legion_field_allocator_allocate_field(
                            allocator, sizeof(double), FID_FIELD_A)
    assert(fida == FID_FIELD_A)
    var fidb = Lg.legion_field_allocator_allocate_field(
                            allocator, sizeof(int), FID_FIELD_B)
    assert(fidb == FID_FIELD_B)
    C.printf("Allocated two fields with Field IDs %d and %d\n", fida, fidb)
    Lg.legion_field_allocator_destroy(allocator)
  end

  -- LogicalRegion unstructured_lr = 
  --   runtime->create_logical_region(ctx, unstructured_is, fs);
  -- printf("Created unstructured logical region (%x,%x,%x)\n",
  --     unstructured_lr.get_index_space().id, 
  --     unstructured_lr.get_field_space().get_id(),
  --     unstructured_lr.get_tree_id());
  -- LogicalRegion structured_lr = 
  --   runtime->create_logical_region(ctx, structured_is, fs);
  -- printf("Created structured logical region (%x,%x,%x)\n",
  --     structured_lr.get_index_space().id, 
  --     structured_lr.get_field_space().get_id(),
  --     structured_lr.get_tree_id());
  var unstructured_lr =
    Lg.legion_logical_region_create(runtime, ctx, unstructured_is, fs)
  C.printf("Created unstructured logical region (%x,%x,%x)\n",
      unstructured_lr.index_space.id,
      unstructured_lr.field_space.id,
      unstructured_lr.tree_id)
  var structured_lr =
    Lg.legion_logical_region_create(runtime, ctx, structured_is, fs)
  C.printf("Created structured logical region (%x,%x,%x)\n",
      structured_lr.index_space.id,
      structured_lr.field_space.id,
      structured_lr.tree_id)
  
  -- LogicalRegion no_clone_lr =
  --   runtime->create_logical_region(ctx, structured_is, fs);
  -- assert(structured_lr.get_tree_id() != no_clone_lr.get_tree_id());
  var no_clone_lr =
    Lg.legion_logical_region_create(runtime, ctx, structured_is, fs)
  assert(structured_lr.tree_id ~= no_clone_lr.tree_id)

  -- runtime->destroy_logical_region(ctx, unstructured_lr);
  -- runtime->destroy_logical_region(ctx, structured_lr);
  -- runtime->destroy_logical_region(ctx, no_clone_lr);
  -- runtime->destroy_field_space(ctx, fs);
  -- runtime->destroy_index_space(ctx, unstructured_is);
  -- runtime->destroy_index_space(ctx, structured_is);
  -- printf("Successfully cleaned up all of our resources\n");
  Lg.legion_logical_region_destroy(runtime, ctx, unstructured_lr)
  Lg.legion_logical_region_destroy(runtime, ctx, structured_lr)
  Lg.legion_logical_region_destroy(runtime, ctx, no_clone_lr)
  Lg.legion_field_space_destroy(runtime, ctx, fs)
  Lg.legion_index_space_destroy(runtime, ctx, unstructured_is)
  Lg.legion_index_space_destroy(runtime, ctx, structured_is)
  C.printf("Successfully cleaned up all of our resources\n")
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