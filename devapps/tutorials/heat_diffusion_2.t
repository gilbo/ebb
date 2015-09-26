import "compiler.liszt"

local ioOff = L.require 'domains.ioOff'
local PN    = L.require 'lib.pathname'
local cmath = terralib.includecstring '#include <math.h>'

local tri_mesh_filename = PN.scriptdir() .. 'bunny.off'
local bunny = ioOff.LoadTrimesh(tri_mesh_filename)

-- Let's try writing the previous heat diffusion example a couple of
-- different ways.  We'll see some ways that don't work and
-- some that do.  In the process we'll learn a bit more about
-- how Liszt works and different ways of defining Liszt simulations.

------------------------------------------------------------------------------

local timestep = L.Global(L.double, 0.45)
local conduction = L.Constant(L.double, 1.0)

bunny.vertices:NewField('temperature', L.double)

local function initial_temperature(vertex_index)
  if vertex_index == 0 then return 3000.0
                       else return 0.0 end
end
bunny.vertices.temperature:Load(initial_temperature)

bunny.vertices:NewField('d_temperature', L.double):Load(0.0)

------------------------------------------------------------------------------

-- To start with, let's re-define the original 
-- computation's functions.

local liszt compute_update ( v : bunny.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.temperature
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.temperature
  v.d_temperature = timestep * conduction * diff_t
end

local liszt apply_update ( v : bunny.vertices )
  v.temperature += v.d_temperature
end

------------------------------------------------------------------------------

-- Let's take a second to talk about how Liszt compiles.  As you may have
-- noticed, Liszt is a Lua-embedded DSL.  That means that the bulk of this
-- file is just Lua code.  When we define a Liszt function using the
-- `liszt` command, then all of a sudden we switch from Lua into
-- Liszt code.  Instead of declaring variables with `local` we use
-- `var`, and a number of other changes (e.g. explicitly typed variables)

-- Lua is an interpreted and dynamically typed language,
-- which means each line of the program is executed as you encounter it
-- and any type-errors won't be discovered until you execute the code.

-- Liszt is a compiled and statically typed language,
-- which means that Liszt functions are passed through a compiler (LLVM)
-- in order to generate high performance code.  Also, any type errors
-- in Liszt code will be reported when its type-checked and then all
-- of then we know the compiled version won't have any trivial type
-- errors in it.

-- How the heck do these two languages interact?

-- The basic idea is that Liszt is JiT (Just in Time) compiled using a
-- two-stage model (similar to Terra)
--    Stage 1: (aka. Definition-Time)
--        Whenever the thread of control in the Lua interpreter
--        reaches a Liszt function declaration, *specialization* occurs.
--    Stage 2: (aka. Compile-Time)
--        Whenever the function is first used, we compile it.
--        At this point, *typechecking*, and *compilation* occur

-- *specialization* is when we expand any Lua values from the surrounding
-- environment and capture their values.  This happens at Definition-Time.

-- *typechecking* is when we check that all of the types are valid.
-- We also expand macros at this point (advanced feature) and
-- run checks to prevent parallel races

-- *compilation* is generating the specialized code to execute a function with


-- So, when the thread of control of the Lua code reaches this point
-- we will have run *specialization* on `compute_update` and `apply_update`.
-- BUT we haven't run typechecking on either function, and we haven't
-- compiled either function.

-- Let's go ahead and run the simulation forward for one time-step
bunny.vertices:foreach(compute_update)
bunny.vertices:foreach(apply_update)

-- Now, typechecking and compilation have happened for
-- both `compute_update` and for `apply_update`.

-- Great, now that we've got that straight, let's look at
-- how Liszt prevents parallel races using the typechecker.

------------------------------------------------------------------------------

-- Liszt functions can call each other (non-recursively).
-- So we might rationally think
--  "Hmmm, why are we mapping two functions,
--    when we could be mapping just one?"
-- Good question.  Let's try defining an aggregate function

local liszt compute_and_apply_update( v : bunny.vertices )
  compute_update(v)
  apply_update(v)
end

-- Remember, at this point Liszt hasn't yet tried to typecheck
-- `compute_and_apply_update`.  To see what happens, uncomment the
-- following line.

--bunny.vertices:foreach(compute_and_apply_update)

-- You should see a big error.  If you look up above the stack dump,
-- you'll see an error that looks something like this:
--
--    heat_diffusion_2.t:50: REDUCE(+) Phase is incompatible with
--    heat_diffusion_2.t:45: READ Phase
--    
--      v.temperature += v.d_temperature
--                     ^
--
-- The typechecker / phasechecker is telling you that there's a
-- Phase conflict in `compute_and_apply_update()` between trying
-- to READ from the `vertices.temperature` field and trying to
-- REDUCE values into it.
-- 
-- Remember how we said that you can think of these functions as
-- defining the body of a parallel for loop?  Well, what would happen
-- if we ran a parallel loop with a body that tried to READ and REDUCE
-- the same values concurrently?  The result would be undefined, because
-- the parallel computation might happen in different orders.  This is
-- what we would normally call a race condition.

-- Liszt prevents programmers from running code with potential races
-- regardless of whether the programmer is running code on their
-- laptop's CPU, a GPU or a supercomputer.  As a result, Liszt can
-- guarantee that code developed on one machine is safe to port to
-- a different, parallel machine without introducing race conditions.

-- Inside a Liszt function, each Field and Global variable is accessed
-- in some combination of the 3 basic Phases:
--    * READ
--    * WRITE
--    * REDUCE(operator)   (e.g. operator = + or *)
--
-- Additionally, Liszt will notice if all Field accesses are done
-- through the "centered" element.  The element passed in as the main
-- argument of a Liszt function; the element from the relation that
-- the function is being mapped over is called the centered element,
-- the centered key or the centered row.  If a field is accessed
-- exclusively through this centered element, then we say the field
-- is being accessed in an EXCLUSIVE phase.

-- The rules for valid phases are then very simple to state
--   * If accessed with EXCLUSIVE phase, a function can READ/WRITE/REDUCE
--        as much as it wants to.  We're guaranteed exclusive access.
--   * If accessed with non-EXCLUSIVE phase, then all field accesses must
--        either be READ or REDUCE(op) with consistent use of the same op
--   * Globals are treated as Fields that can never be accessed EXCLUSIVE-ly

-- So now we understand why the computation uses two functions
-- instead of one.  We need to use some kind of buffering trick
-- in order to execute safely on parallel machines.  If you stop and
-- think about this, it becomes clear that this is a fundamental
-- constraint of parallel processing that Liszt is exposing to you.

------------------------------------------------------------------------------

-- We solved this problem by creating an change buffer and then using it
-- to update the original data.  However, another good solution would have
-- been to keep two copies of the temperature field around and repeatedly
-- update one of these from the either.

-- This solution would normally involve a lot of bookkeeping, but since
-- it's so common of a pattern, Liszt provides us with some helpful
-- functions to make it easier to write.

-- First, let's create our secondary buffer of temperatures
bunny.vertices:NewField('temp_temperature', L.double):Load(0.0)

-- Now, we'll define the single compute function
local liszt compute_with_temp ( v : bunny.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.temperature
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.temperature
  -- *** Notice that the following line has changed ***
  v.temp_temperature = v.temperature + timestep * conduction * diff_t
  -- ***
end

-- Let's go ahead and run a single iteration of this new scheme
bunny.vertices:foreach(compute_with_temp)

-- Now, rather than trying to ping-pong the computation back and forth,
-- let's just swap the underlying buffers.  This just requires a pointer
-- swap, and so there's no cost to performing the operation.
bunny.vertices:Swap('temperature', 'temp_temperature')

-- Great, now we can go ahead and take another step
bunny.vertices:foreach(compute_with_temp)
bunny.vertices:Swap('temperature', 'temp_temperature')


------------------------------------------------------------------------------


-- Another variation of the basic heat diffusion works by mapping our
-- computation over the edges rather than the vertices of the underlying
-- relation.

-- First, let's define the degree of each vertex.  We had been computing
-- that on the fly before, but it'll be easier to cache it now.
bunny.vertices:NewField('degree', L.int):Load(0)

local liszt compute_degree ( v : bunny.vertices )
  for e in v.edges do
    v.degree += 1
  end
end

bunny.vertices:foreach(compute_degree)

-- Great.  Now, let's define an edge-centered version of heat diffusion.
-- At this point we should note that Trimesh provides directed edges,
-- so we'll get one computation for each edge direction.

local liszt compute_edge_update ( e : bunny.edges )
  var ht = e.head.temperature
  var tt = e.tail.temperature
  var dt = ht - tt

  var frac_t = dt / e.tail.degree
  e.tail.d_temperature += timestep * conduction * frac_t
end

-- In this version, we're going to reduce into the `d_temperature` field.
-- In effect, we've re-implemented the heat diffusion as a scatter
-- rather than a gather computation.

-- We'll need some way to zero out the `d_temperature` field
local liszt zero_d_temperature ( v : bunny.vertices )
  v.d_temperature = 0
end

-- Then, we can define a way to apply the update.  For convenience,
-- we'll go ahead and zero out the temperature after we read it.
local liszt apply_edge_update ( v : bunny.vertices )
  v.temperature += v.d_temperature
  zero_d_temperature(v)
end

-- To execute this version of heat diffusion, we'll first want to
-- zero out the temperature field and then we'll run the compute_update
-- and apply_update steps one after another.

bunny.vertices:foreach(zero_d_temperature)

bunny.edges:foreach(compute_edge_update)
bunny.vertices:foreach(apply_edge_update)

bunny.edges:foreach(compute_edge_update)
bunny.vertices:foreach(apply_edge_update)

-- and on and on and on ...


------------------------------------------------------------------------------


-- Great, let's summarize what we learned in this tutorial real quick

-- 1) We learned about Liszt's execution model
--      - Liszt specializes code at Definition-Time
--      - Liszt typechecks code at Compile-Time
--      - Liszt waits until the first time
--          a function is executed to compile it

-- 2) We learned about how Liszt uses Phases to ensure safe and portable
--      parallel code.

-- 3) We learned that we can `Swap` fields, which can help us manage
--      intermediate parallel fields more easily.

-- 4) We learned how to transform gather-style computations into
--      scatter-style reductions.




